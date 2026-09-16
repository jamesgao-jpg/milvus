package searchutil

import (
	"io"
	"net"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestSearchBenchmarkTrackedConnectionCounters(t *testing.T) {
	tracker := newSearchBenchmarkConnectionTracker()
	trackedSide, peerSide := net.Pipe()
	tracked := tracker.track(trackedSide)
	t.Cleanup(func() {
		tracked.Close()
		peerSide.Close()
	})

	remote := trackedSide.RemoteAddr().String()
	start := tracker.snapshot(remote)
	payload := []byte("benchmark-payload")

	writeDone := make(chan error, 1)
	go func() {
		_, err := tracked.Write(payload)
		writeDone <- err
	}()
	received := make([]byte, len(payload))
	_, err := io.ReadFull(peerSide, received)
	require.NoError(t, err)
	require.NoError(t, <-writeDone)
	require.Equal(t, payload, received)

	go func() {
		_, err := peerSide.Write(payload)
		writeDone <- err
	}()
	_, err = io.ReadFull(tracked, received)
	require.NoError(t, err)
	require.NoError(t, <-writeDone)

	delta := tracker.snapshot(remote).since(start)
	require.Equal(t, uint64(len(payload)), delta.connectionReadBytes)
	require.Equal(t, uint64(len(payload)), delta.connectionWriteBytes)
	require.False(t, delta.tcpInfoAvailable)
}
