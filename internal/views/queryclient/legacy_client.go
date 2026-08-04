package queryclient

import (
	"context"
	"errors"
	"io"
	"sync"

	"golang.org/x/sync/errgroup"

	"github.com/milvus-io/milvus/internal/util/searchutil"
	"github.com/milvus-io/milvus/internal/views/queryclient/resolver"
	"github.com/milvus-io/milvus/internal/views/qviews"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/viewpb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// Client exposes QueryView query client domains.
type Client interface {
	Legacy() LegacyClient
}

// LegacyClient returns batch results or the final iterator ReduceStream.
type LegacyClient interface {
	Search(ctx context.Context, req *LegacySearchRequest) (*LegacySearchResult, error)
	Query(ctx context.Context, req *LegacyQueryRequest) (*LegacyQueryResult, error)
}

type LegacySearchRequest struct {
	Req *internalpb.SearchRequest
}

type LegacySearchResult struct {
	Results []*internalpb.SearchResults
	Stream  searchutil.ReduceStream
	Plans   []ShardPlan
}

type LegacyQueryRequest struct {
	Req *internalpb.RetrieveRequest
}

type LegacyQueryResult struct {
	Results []*internalpb.RetrieveResults
	Plans   []ShardPlan
}

type legacyOnlyClient struct {
	legacy LegacyClient
}

func (c *legacyOnlyClient) Legacy() LegacyClient {
	return c.legacy
}

type legacyClient struct {
	shardClient   *shardViewQueryClient
	shardResolver resolver.ShardResolver
}

type prefetchedReduceStream struct {
	stream     searchutil.ReduceStream
	firstChunk *internalpb.SearchResults
}

func (s *prefetchedReduceStream) Recv() (*internalpb.SearchResults, error) {
	if s.firstChunk != nil {
		chunk := s.firstChunk
		s.firstChunk = nil
		return chunk, nil
	}
	return s.stream.Recv()
}

func (s *prefetchedReduceStream) Close() error {
	s.firstChunk = nil
	return s.stream.Close()
}

func (s *prefetchedReduceStream) Interrupt() (*internalpb.SearchResults, error) {
	s.firstChunk = nil
	return s.stream.Interrupt()
}

func NewLegacyViewQueryClient(
	cfg ViewQueryClientConfig,
	queryPlanClient QueryPlanClient,
	queryServiceClient ViewQueryServiceClient,
	shardResolver resolver.ShardResolver,
	replicaPicker ReplicaPicker,
) Client {
	return &legacyOnlyClient{
		legacy: newLegacyClient(cfg, queryPlanClient, queryServiceClient, shardResolver, replicaPicker),
	}
}

func newLegacyClient(
	cfg ViewQueryClientConfig,
	queryPlanClient QueryPlanClient,
	queryServiceClient ViewQueryServiceClient,
	shardResolver resolver.ShardResolver,
	replicaPicker ReplicaPicker,
) *legacyClient {
	if cfg.MaxRetries <= 0 {
		cfg.MaxRetries = defaultMaxRetries
	}
	return &legacyClient{
		shardClient:   newShardViewQueryClient(cfg.MaxRetries, queryPlanClient, queryServiceClient, shardResolver, replicaPicker),
		shardResolver: shardResolver,
	}
}

func (c *legacyClient) Search(ctx context.Context, req *LegacySearchRequest) (*LegacySearchResult, error) {
	if req.Req.GetIsIterator() &&
		!req.Req.GetIsAdvanced() &&
		len(req.Req.GetSubReqs()) == 0 &&
		req.Req.GetGroupByFieldId() <= 0 &&
		len(req.Req.GetGroupByFieldIds()) == 0 {
		var lastErr error
		for attempt := 0; attempt < c.shardClient.maxRetries; attempt++ {
			vchannels, err := c.shardResolver.ResolveVChannels(ctx, req.Req.GetCollectionID())
			if err != nil {
				return nil, err
			}

			vchannelStreams := make([]searchutil.ReduceStream, len(vchannels))
			shardPlans := make([]ShardPlan, len(vchannels))
			var g errgroup.Group
			for i := range vchannels {
				i := i
				g.Go(func() error {
					stream, plan, err := c.shardClient.SearchStream(ctx, vchannels[i], req.Req)
					if err != nil {
						return err
					}
					vchannelStreams[i] = stream
					shardPlans[i] = *plan
					return nil
				})
			}

			if err := g.Wait(); err != nil {
				for _, stream := range vchannelStreams {
					if stream != nil {
						err = errors.Join(err, stream.Close())
					}
				}
				if ctx.Err() != nil {
					return nil, ctx.Err()
				}
				lastErr = err
				continue
			}

			finalStream, err := searchutil.NewReduceStream(req.Req, vchannelStreams, defaultSearchStreamChunkSize)
			if err != nil {
				for _, stream := range vchannelStreams {
					err = errors.Join(err, stream.Close())
				}
				return nil, err
			}

			firstChunk, recvErr := finalStream.Recv()
			if recvErr != nil && !errors.Is(recvErr, io.EOF) {
				err = errors.Join(recvErr, finalStream.Close())
				if ctx.Err() != nil {
					return nil, ctx.Err()
				}
				lastErr = err
				continue
			}

			stream := finalStream
			if firstChunk != nil {
				stream = &prefetchedReduceStream{
					stream:     finalStream,
					firstChunk: firstChunk,
				}
			}

			return &LegacySearchResult{
				Stream: stream,
				Plans:  shardPlans,
			}, nil
		}
		return nil, lastErr
	}

	vchannels, err := c.shardResolver.ResolveVChannels(ctx, req.Req.CollectionID)
	if err != nil {
		return nil, err
	}

	collector := newLegacySearchCollector()
	shardPlans := make([]ShardPlan, len(vchannels))
	g, gCtx := errgroup.WithContext(ctx)
	for i := range vchannels {
		i := i
		g.Go(func() error {
			plan, err := c.shardClient.Search(gCtx, &ShardSearchRequest{
				VChannel: vchannels[i],
				Req:      req.Req,
				Reducer:  collector,
			})
			if err != nil {
				return err
			}
			shardPlans[i] = *plan
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return nil, err
	}
	return &LegacySearchResult{
		Results: collector.Results(),
		Plans:   shardPlans,
	}, nil
}

func (c *legacyClient) Query(ctx context.Context, req *LegacyQueryRequest) (*LegacyQueryResult, error) {
	vchannels, err := c.shardResolver.ResolveVChannels(ctx, req.Req.CollectionID)
	if err != nil {
		return nil, err
	}

	collector := newLegacyQueryCollector()
	shardPlans := make([]ShardPlan, len(vchannels))
	g, gCtx := errgroup.WithContext(ctx)
	for i := range vchannels {
		i := i
		g.Go(func() error {
			plan, err := c.shardClient.Query(gCtx, &ShardQueryRequest{
				VChannel: vchannels[i],
				Req:      req.Req,
				Reducer:  collector,
			})
			if err != nil {
				return err
			}
			shardPlans[i] = *plan
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return nil, err
	}
	return &LegacyQueryResult{
		Results: collector.Results(),
		Plans:   shardPlans,
	}, nil
}

type legacySearchCollector struct {
	mu      sync.Mutex
	results map[string][]*internalpb.SearchResults
}

func newLegacySearchCollector() *legacySearchCollector {
	return &legacySearchCollector{
		results: make(map[string][]*internalpb.SearchResults),
	}
}

func (c *legacySearchCollector) Add(shardID qviews.ShardID, resp *viewpb.SearchOnViewResponse) error {
	result := resp.GetLegacyResults()
	if result == nil {
		return merr.WrapErrServiceInternalMsg("missing legacy search result for shard %s", shardID.String())
	}
	if !merr.Ok(result.GetStatus()) {
		return merr.Error(result.GetStatus())
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	c.results[shardID.String()] = append(c.results[shardID.String()], result)
	return nil
}

func (c *legacySearchCollector) ResetShard(shardID qviews.ShardID) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.results, shardID.String())
}

func (c *legacySearchCollector) Finish() (*internalpb.SearchResults, error) {
	return nil, merr.WrapErrServiceInternalMsg("legacy search collector does not reduce results")
}

func (c *legacySearchCollector) Results() []*internalpb.SearchResults {
	c.mu.Lock()
	defer c.mu.Unlock()

	results := make([]*internalpb.SearchResults, 0)
	for _, shardResults := range c.results {
		results = append(results, shardResults...)
	}
	return results
}

type legacyQueryCollector struct {
	mu      sync.Mutex
	results map[string][]*internalpb.RetrieveResults
}

func newLegacyQueryCollector() *legacyQueryCollector {
	return &legacyQueryCollector{
		results: make(map[string][]*internalpb.RetrieveResults),
	}
}

func (c *legacyQueryCollector) Add(shardID qviews.ShardID, resp *viewpb.QueryOnViewResponse) error {
	result := resp.GetLegacyResults()
	if result == nil {
		return merr.WrapErrServiceInternalMsg("missing legacy query result for shard %s", shardID.String())
	}
	if !merr.Ok(result.GetStatus()) {
		return merr.Error(result.GetStatus())
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	c.results[shardID.String()] = append(c.results[shardID.String()], result)
	return nil
}

func (c *legacyQueryCollector) ResetShard(shardID qviews.ShardID) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.results, shardID.String())
}

func (c *legacyQueryCollector) Finish() (*internalpb.RetrieveResults, error) {
	return nil, merr.WrapErrServiceInternalMsg("legacy query collector does not reduce results")
}

func (c *legacyQueryCollector) Results() []*internalpb.RetrieveResults {
	c.mu.Lock()
	defer c.mu.Unlock()

	results := make([]*internalpb.RetrieveResults, 0)
	for _, shardResults := range c.results {
		results = append(results, shardResults...)
	}
	return results
}
