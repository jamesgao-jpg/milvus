package handler

import (
	"context"
	"sync"

	"github.com/milvus-io/milvus/internal/util/queryutil"
	"github.com/milvus-io/milvus/internal/util/searchutil"
	"github.com/milvus-io/milvus/internal/util/streamingutil/service/attributes"
	"github.com/milvus-io/milvus/internal/util/streamingutil/service/contextutil"
	"github.com/milvus-io/milvus/internal/util/streamingutil/service/discoverer"
	"github.com/milvus-io/milvus/internal/util/streamingutil/service/lazygrpc"
	"github.com/milvus-io/milvus/internal/views/viewerror"
	"github.com/milvus-io/milvus/pkg/v3/mlog"
	"github.com/milvus-io/milvus/pkg/v3/proto/viewpb"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/typeutil"
)

// QueryViewClient is the QueryView domain client under the QueryNode handler client.
type QueryViewClient interface {
	SearchOnView(ctx context.Context, nodeID int64, req *viewpb.SearchOnViewRequest) (*viewpb.SearchOnViewResponse, error)
	SearchOnViewStream(ctx context.Context, nodeID int64, req *viewpb.SearchOnViewRequest) (searchutil.ReduceStream, error)
	QueryOnView(ctx context.Context, nodeID int64, req *viewpb.QueryOnViewRequest) (*viewpb.QueryOnViewResponse, error)
	QueryOnViewStream(ctx context.Context, nodeID int64, req *viewpb.QueryOnViewRequest) (queryutil.ReduceStream, error)
	RequeryOnView(ctx context.Context, nodeID int64, req *viewpb.RequeryOnViewRequest) (*viewpb.RequeryOnViewResponse, error)
}

type queryViewClient struct {
	owner           *clientImpl
	service         lazygrpc.Service[viewpb.ViewQueryServiceClient]
	reusableQueries *queryutil.ReusableQueryStreamPool
	stopReusable    context.CancelFunc
	reusableWatch   sync.WaitGroup
}

func newQueryViewClient(owner *clientImpl, conn lazygrpc.Conn) *queryViewClient {
	client := &queryViewClient{
		owner:   owner,
		service: lazygrpc.WithServiceCreator(conn, viewpb.NewViewQueryServiceClient),
	}
	if paramtable.Get().ProxyCfg.EnableReduceStream.GetAsBool() &&
		paramtable.Get().ProxyCfg.EnableReusableQueryStream.GetAsBool() {
		ctx, cancel := context.WithCancel(context.Background())
		client.stopReusable = cancel
		client.reusableQueries = queryutil.NewReusableQueryStreamPool(ctx, paramtable.Get().ProxyCfg.ReusableQueryStreamPoolSize.GetAsInt())
		if owner.rb != nil {
			client.reusableWatch.Add(1)
			go func() {
				defer client.reusableWatch.Done()
				err := owner.rb.Resolver().Watch(ctx, func(state discoverer.VersionedState) error {
					nodes := make(map[int64]struct{}, len(state.State.Addresses))
					for _, address := range state.State.Addresses {
						if address.Attributes != nil {
							if id := attributes.GetServerID(address.Attributes); id != nil {
								nodes[*id] = struct{}{}
							}
						}
					}
					client.reusableQueries.RetainNodes(nodes)
					return nil
				})
				if err != nil && ctx.Err() == nil {
					mlog.Warn(ctx, "reusable Query RPC discovery watch stopped", mlog.Err(err))
				}
			}()
		}
		mlog.Info(ctx, "benchmark-only reusable Query RPCs enabled")
	}
	return client
}

func (qvc *queryViewClient) SearchOnView(ctx context.Context, nodeID int64, req *viewpb.SearchOnViewRequest) (*viewpb.SearchOnViewResponse, error) {
	return executeViewQueryRPC(ctx, qvc, nodeID, "ViewQueryService.SearchOnView", func(ctx context.Context, client viewpb.ViewQueryServiceClient) (*viewpb.SearchOnViewResponse, error) {
		return client.SearchOnView(ctx, req)
	})
}

func (qvc *queryViewClient) SearchOnViewStream(ctx context.Context, nodeID int64, req *viewpb.SearchOnViewRequest) (searchutil.ReduceStream, error) {
	return executeViewQueryRPC(ctx, qvc, nodeID, "ViewQueryService.SearchOnViewStream", func(ctx context.Context, client viewpb.ViewQueryServiceClient) (searchutil.ReduceStream, error) {
		return searchutil.NewGRPCReduceStream(ctx, client, req)
	})
}

func (qvc *queryViewClient) QueryOnView(ctx context.Context, nodeID int64, req *viewpb.QueryOnViewRequest) (*viewpb.QueryOnViewResponse, error) {
	return executeViewQueryRPC(ctx, qvc, nodeID, "ViewQueryService.QueryOnView", func(ctx context.Context, client viewpb.ViewQueryServiceClient) (*viewpb.QueryOnViewResponse, error) {
		return client.QueryOnView(ctx, req)
	})
}

func (qvc *queryViewClient) QueryOnViewStream(ctx context.Context, nodeID int64, req *viewpb.QueryOnViewRequest) (queryutil.ReduceStream, error) {
	return executeViewQueryRPC(ctx, qvc, nodeID, "ViewQueryService.QueryOnViewStream", func(ctx context.Context, client viewpb.ViewQueryServiceClient) (queryutil.ReduceStream, error) {
		if qvc.reusableQueries != nil {
			stream, err := qvc.reusableQueries.NewStream(ctx, client, nodeID, req)
			if stream != nil || err != nil {
				return stream, err
			}
		}
		return queryutil.NewGRPCReduceStream(ctx, client, req)
	})
}

func (qvc *queryViewClient) RequeryOnView(ctx context.Context, nodeID int64, req *viewpb.RequeryOnViewRequest) (*viewpb.RequeryOnViewResponse, error) {
	return executeViewQueryRPC(ctx, qvc, nodeID, "ViewQueryService.RequeryOnView", func(ctx context.Context, client viewpb.ViewQueryServiceClient) (*viewpb.RequeryOnViewResponse, error) {
		return client.RequeryOnView(ctx, req)
	})
}

func executeViewQueryRPC[T any](
	ctx context.Context,
	qvc *queryViewClient,
	nodeID int64,
	method string,
	call func(context.Context, viewpb.ViewQueryServiceClient) (T, error),
) (T, error) {
	if !qvc.owner.lifetime.Add(typeutil.LifetimeStateWorking) {
		var zero T
		return zero, viewerror.NewOnShutdownError("querynode client is closing")
	}
	defer qvc.owner.lifetime.Done()

	client, err := qvc.service.GetService(ctx)
	if err != nil {
		var zero T
		return zero, err
	}
	resp, err := call(contextutil.WithPickServerID(ctx, nodeID), client)
	return resp, viewerror.ConvertViewError(method, err)
}

func (qvc *queryViewClient) close() {
	qvc.stopReusableQueries()
	qvc.service.Close()
}

// stopReusableQueries aborts transport before client shutdown waits for open calls.
func (qvc *queryViewClient) stopReusableQueries() {
	if qvc.stopReusable != nil {
		qvc.stopReusable()
	}
	if qvc.reusableQueries != nil {
		qvc.reusableQueries.Close()
	}
	qvc.reusableWatch.Wait()
}
