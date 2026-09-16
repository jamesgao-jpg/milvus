//go:build linux

package searchutil

import (
	"net"
	"syscall"

	"golang.org/x/sys/unix"
)

type searchBenchmarkTCPSnapshot struct {
	bytesReceived, bytesSent, bytesAcked, notSentBytes uint64
	available                                          bool
}

func readSearchBenchmarkTCPInfo(connection net.Conn) (searchBenchmarkTCPSnapshot, bool) {
	syscallConnection, ok := connection.(syscall.Conn)
	if !ok {
		return searchBenchmarkTCPSnapshot{}, false
	}
	rawConnection, err := syscallConnection.SyscallConn()
	if err != nil {
		return searchBenchmarkTCPSnapshot{}, false
	}
	var info *unix.TCPInfo
	var controlError error
	if err := rawConnection.Control(func(fd uintptr) {
		info, controlError = unix.GetsockoptTCPInfo(int(fd), unix.IPPROTO_TCP, unix.TCP_INFO)
	}); err != nil || controlError != nil || info == nil {
		return searchBenchmarkTCPSnapshot{}, false
	}
	return searchBenchmarkTCPSnapshot{
		bytesReceived: info.Bytes_received,
		bytesSent:     info.Bytes_sent,
		bytesAcked:    info.Bytes_acked,
		notSentBytes:  uint64(info.Notsent_bytes),
		available:     true,
	}, true
}
