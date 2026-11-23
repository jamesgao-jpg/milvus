// Licensed to the LF AI & Data foundation under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package reducestream

import (
	"context"
	"io"

	"github.com/cockroachdb/errors"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/querypb"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
)

// OpenWorkerSearchStream checks the worker's first response before returning it
// to the reduction. An older worker can still provide one complete unary result.
func OpenWorkerSearchStream(ctx context.Context, client querypb.QueryNodeClient, req *querypb.SearchRequest, chunkBytes int32) (ReduceStream, error) {
	rpcCtx, cancel := context.WithCancel(ctx)
	rpc, err := client.SearchSegmentsStream(rpcCtx)
	if err == nil {
		err = rpc.Send(&querypb.SearchStreamRequest{Request: req, ChunkBytes: chunkBytes})
	}
	if err == nil {
		err = rpc.CloseSend()
	}
	var first *internalpb.SearchResults
	if err == nil || (rpc != nil && errors.Is(err, io.EOF)) {
		first, err = rpc.Recv()
	}
	if status.Code(err) == codes.Unimplemented {
		cancel()
		rpc = nil
		first, err = client.SearchSegments(ctx, req)
	}
	if err == nil {
		if first == nil {
			err = merr.WrapErrServiceInternalMsg("worker SearchStream returned no result")
		} else {
			err = merr.Error(first.GetStatus())
		}
	} else if errors.Is(err, io.EOF) {
		err = merr.WrapErrServiceInternalMsg("worker SearchStream ended before its first Chunk")
	}
	if err != nil {
		cancel()
		return nil, err
	}
	return &grpcReduceStream{rpc: rpc, first: first, cancel: cancel}, nil
}

type grpcReduceStream struct {
	rpc    querypb.QueryNode_SearchSegmentsStreamClient
	first  *internalpb.SearchResults
	cancel context.CancelFunc
}

func (s *grpcReduceStream) Recv() (*internalpb.SearchResults, error) {
	if s.first != nil {
		result := s.first
		s.first = nil
		return result, nil
	}
	if s.rpc == nil {
		return nil, io.EOF
	}
	return s.rpc.Recv()
}

func (s *grpcReduceStream) Close() error {
	s.cancel()
	return nil
}

func (s *grpcReduceStream) Interrupt() (*internalpb.SearchResults, error) {
	return nil, merr.ErrServiceUnimplemented
}
