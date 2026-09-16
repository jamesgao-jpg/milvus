package searchutil

import (
	"context"
	"net"
	"sync"
	"sync/atomic"
	"syscall"
)

type searchBenchmarkTransportSnapshot struct {
	connectionReadBytes  uint64
	connectionWriteBytes uint64
	tcpBytesReceived     uint64
	tcpBytesSent         uint64
	tcpBytesAcked        uint64
	tcpNotSentBytes      uint64
	tcpInfoAvailable     bool
}

func (end searchBenchmarkTransportSnapshot) since(start searchBenchmarkTransportSnapshot) searchBenchmarkTransportSnapshot {
	delta := searchBenchmarkTransportSnapshot{
		connectionReadBytes:  counterDelta(end.connectionReadBytes, start.connectionReadBytes),
		connectionWriteBytes: counterDelta(end.connectionWriteBytes, start.connectionWriteBytes),
		tcpNotSentBytes:      end.tcpNotSentBytes,
	}
	if !start.tcpInfoAvailable || !end.tcpInfoAvailable {
		return delta
	}
	delta.tcpBytesReceived = counterDelta(end.tcpBytesReceived, start.tcpBytesReceived)
	delta.tcpBytesSent = counterDelta(end.tcpBytesSent, start.tcpBytesSent)
	delta.tcpBytesAcked = counterDelta(end.tcpBytesAcked, start.tcpBytesAcked)
	delta.tcpInfoAvailable = true
	return delta
}

func counterDelta(end, start uint64) uint64 {
	if end < start {
		return 0
	}
	return end - start
}

func (s *searchBenchmarkTransportSnapshot) add(other searchBenchmarkTransportSnapshot) {
	s.connectionReadBytes += other.connectionReadBytes
	s.connectionWriteBytes += other.connectionWriteBytes
	s.tcpBytesReceived += other.tcpBytesReceived
	s.tcpBytesSent += other.tcpBytesSent
	s.tcpBytesAcked += other.tcpBytesAcked
	s.tcpNotSentBytes += other.tcpNotSentBytes
}

type searchBenchmarkTrackedConn struct {
	net.Conn
	readBytes  atomic.Uint64
	writeBytes atomic.Uint64
	closeOnce  sync.Once
	closeMu    sync.Mutex
	finalTCP   searchBenchmarkTCPSnapshot
}

func (c *searchBenchmarkTrackedConn) Read(buffer []byte) (int, error) {
	n, err := c.Conn.Read(buffer)
	c.readBytes.Add(uint64(n))
	return n, err
}

func (c *searchBenchmarkTrackedConn) Write(buffer []byte) (int, error) {
	n, err := c.Conn.Write(buffer)
	c.writeBytes.Add(uint64(n))
	return n, err
}

func (c *searchBenchmarkTrackedConn) Close() error {
	c.closeOnce.Do(func() {
		c.closeMu.Lock()
		c.finalTCP, _ = readSearchBenchmarkTCPInfo(c.Conn)
		c.closeMu.Unlock()
	})
	return c.Conn.Close()
}

func (c *searchBenchmarkTrackedConn) SyscallConn() (syscall.RawConn, error) {
	connection, ok := c.Conn.(syscall.Conn)
	if !ok {
		return nil, syscall.EINVAL
	}
	return connection.SyscallConn()
}

func (c *searchBenchmarkTrackedConn) snapshot() searchBenchmarkTransportSnapshot {
	tcp, available := readSearchBenchmarkTCPInfo(c.Conn)
	if !available {
		c.closeMu.Lock()
		tcp = c.finalTCP
		c.closeMu.Unlock()
		available = tcp.available
	}
	return searchBenchmarkTransportSnapshot{
		connectionReadBytes:  c.readBytes.Load(),
		connectionWriteBytes: c.writeBytes.Load(),
		tcpBytesReceived:     tcp.bytesReceived,
		tcpBytesSent:         tcp.bytesSent,
		tcpBytesAcked:        tcp.bytesAcked,
		tcpNotSentBytes:      tcp.notSentBytes,
		tcpInfoAvailable:     available,
	}
}

type searchBenchmarkConnectionTracker struct {
	mu       sync.Mutex
	byRemote map[string][]*searchBenchmarkTrackedConn
}

func newSearchBenchmarkConnectionTracker() *searchBenchmarkConnectionTracker {
	return &searchBenchmarkConnectionTracker{byRemote: make(map[string][]*searchBenchmarkTrackedConn)}
}

func (t *searchBenchmarkConnectionTracker) track(connection net.Conn) net.Conn {
	tracked := &searchBenchmarkTrackedConn{Conn: connection}
	t.mu.Lock()
	t.byRemote[connection.RemoteAddr().String()] = append(t.byRemote[connection.RemoteAddr().String()], tracked)
	t.mu.Unlock()
	return tracked
}

func (t *searchBenchmarkConnectionTracker) snapshot(remoteAddress string) searchBenchmarkTransportSnapshot {
	t.mu.Lock()
	connections := append([]*searchBenchmarkTrackedConn(nil), t.byRemote[remoteAddress]...)
	t.mu.Unlock()
	var snapshot searchBenchmarkTransportSnapshot
	snapshot.tcpInfoAvailable = len(connections) > 0
	for _, connection := range connections {
		connectionSnapshot := connection.snapshot()
		snapshot.add(connectionSnapshot)
		snapshot.tcpInfoAvailable = snapshot.tcpInfoAvailable && connectionSnapshot.tcpInfoAvailable
	}
	return snapshot
}

var (
	searchBenchmarkClientConnections = newSearchBenchmarkConnectionTracker()
	searchBenchmarkServerConnections = newSearchBenchmarkConnectionTracker()
)

type searchBenchmarkTrackedListener struct {
	net.Listener
}

func (l searchBenchmarkTrackedListener) Accept() (net.Conn, error) {
	connection, err := l.Listener.Accept()
	if err != nil {
		return nil, err
	}
	return searchBenchmarkServerConnections.track(connection), nil
}

func TrackSearchBenchmarkListener(listener net.Listener) net.Listener {
	if !SearchBenchmarkMetricsEnabled() {
		return listener
	}
	return searchBenchmarkTrackedListener{Listener: listener}
}

func DialSearchBenchmarkConnection(ctx context.Context, address string) (net.Conn, error) {
	connection, err := (&net.Dialer{}).DialContext(ctx, "tcp", address)
	if err != nil {
		return nil, err
	}
	return searchBenchmarkClientConnections.track(connection), nil
}
