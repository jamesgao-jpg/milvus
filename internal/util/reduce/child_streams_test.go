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

package reduce

import (
	"context"
	"errors"
	"io"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

type cancelableChildStream struct {
	started   chan struct{}
	canceled  chan struct{}
	release   <-chan struct{}
	closeOnce sync.Once
}

// blockingCloseChildStream holds cleanup until every sibling can start closing.
type blockingCloseChildStream struct {
	index   int
	started chan<- int
	release <-chan struct{}
	err     error
	calls   int
}

// Recv makes this cleanup-only fixture immediately drained.
func (s *blockingCloseChildStream) Recv() (int, error) { return 0, io.EOF }

// Close records each invocation and waits for the shared cleanup release.
func (s *blockingCloseChildStream) Close() error {
	s.calls++
	s.started <- s.index
	<-s.release
	return s.err
}

// TestChildStreamsConcurrentClose checks prompt sibling cleanup and stable errors.
func TestChildStreamsConcurrentClose(t *testing.T) {
	for _, count := range []int{0, 1, 4} {
		t.Run(strconv.Itoa(count), func(t *testing.T) {
			started, release := make(chan int, count), make(chan struct{})
			var releaseOnce sync.Once
			unblock := func() { releaseOnce.Do(func() { close(release) }) }
			t.Cleanup(unblock)
			failure := errors.New("child cleanup failed")
			children := make([]*blockingCloseChildStream, count)
			for i := range children {
				children[i] = &blockingCloseChildStream{index: i, started: started, release: release}
				if i%2 == 0 {
					children[i].err = failure
				}
			}
			streams := NewChildStreams[int](children)
			closed := make(chan error, 1)
			go func() { closed <- streams.Close() }()

			// Every child must enter Close while all earlier children remain blocked.
			seen := make(map[int]bool)
			for range count {
				select {
				case index := <-started:
					require.False(t, seen[index])
					seen[index] = true
				case <-time.After(5 * time.Second):
					t.Fatal("not all children started closing before cleanup was released")
				}
			}
			if count > 0 {
				select {
				case <-closed:
					t.Fatal("Close returned before child cleanup finished")
				default:
				}
			}
			unblock()

			// Preserve every indexed failure and join before exposing the result.
			select {
			case err := <-closed:
				if count == 0 {
					require.NoError(t, err)
				} else {
					require.ErrorIs(t, err, failure)
					require.Contains(t, err.Error(), "close child stream 0")
					if count == 4 {
						require.Contains(t, err.Error(), "close child stream 2")
					}
				}
				require.Equal(t, err, streams.Close())
			case <-time.After(5 * time.Second):
				t.Fatal("Close did not join completed child cleanup")
			}
			for _, child := range children {
				require.Equal(t, 1, child.calls)
			}
		})
	}
}

func (s *cancelableChildStream) Recv() (int, error) {
	close(s.started)
	<-s.canceled
	<-s.release
	return 0, context.Canceled
}

func (s *cancelableChildStream) Close() error {
	s.closeOnce.Do(func() { close(s.canceled) })
	return nil
}

func TestChildStreamsCancelAndClose(t *testing.T) {
	for _, mode := range []string{"all", "any"} {
		t.Run(mode, func(t *testing.T) {
			releaseContext, release := context.WithCancel(context.Background())
			defer release()
			child := &cancelableChildStream{
				started: make(chan struct{}), canceled: make(chan struct{}),
				release: releaseContext.Done(),
			}
			streams := NewChildStreams[int]([]ChildStream[int]{child})
			receive := streams.ReceiveUntilReady
			if mode == "any" {
				receive = streams.ReceiveAnyReady
			}
			received := make(chan error, 1)
			go func() {
				received <- receive(func(int) bool { return false }, func(int, int) error {
					t.Error("canceled receive must not accept a Chunk")
					return nil
				})
			}()
			select {
			case <-child.started:
			case <-time.After(5 * time.Second):
				t.Fatal("child Recv did not start")
			}

			streams.Cancel()
			streams.Cancel()
			select {
			case err := <-received:
				require.ErrorIs(t, err, context.Canceled)
			case <-time.After(5 * time.Second):
				t.Fatal("Cancel did not wake the receive wait")
			}
			select {
			case <-child.canceled:
				t.Fatal("Cancel must not close a child before the owner unwinds")
			default:
			}

			closed := make(chan error, 1)
			go func() { closed <- streams.Close() }()
			select {
			case <-child.canceled:
			case <-time.After(5 * time.Second):
				t.Fatal("Close did not cancel the child")
			}
			select {
			case <-closed:
				t.Fatal("Close did not wait for the pending receive")
			default:
			}
			release()
			select {
			case err := <-closed:
				require.NoError(t, err)
			case <-time.After(5 * time.Second):
				t.Fatal("Close did not join the pending receive")
			}
			require.Empty(t, streams.completions)
			require.Equal(t, []bool{false}, streams.receiving)
			require.NoError(t, streams.Close())
			require.ErrorIs(t, receive(func(int) bool { return true }, nil), context.Canceled)
		})
	}
}
