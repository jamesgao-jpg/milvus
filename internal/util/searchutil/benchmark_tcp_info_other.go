//go:build !linux

package searchutil

import "net"

type searchBenchmarkTCPSnapshot struct {
	bytesReceived, bytesSent, bytesAcked, notSentBytes uint64
	available                                          bool
}

func readSearchBenchmarkTCPInfo(net.Conn) (searchBenchmarkTCPSnapshot, bool) {
	return searchBenchmarkTCPSnapshot{}, false
}
