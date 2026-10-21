// Benchmark control tests check capture ownership, local access and complete profile output.
package http

import (
	"compress/gzip"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"runtime/pprof"
	"testing"
	"time"
)

// benchmarkCommand exercises the actual HTTP handler without another data RPC.
func benchmarkCommand(p *benchmarkCPUProfile, action, name, address string) *httptest.ResponseRecorder {
	r := httptest.NewRequest(http.MethodPost, "/debug/benchmark/cpu?action="+action+"&name="+name, nil)
	r.RemoteAddr = address
	w := httptest.NewRecorder()
	p.serveHTTP(w, r)
	return w
}

// TestBenchmarkCPUProfile verifies local control, exclusive ownership and flushed gzip data.
func TestBenchmarkCPUProfile(t *testing.T) {
	p := &benchmarkCPUProfile{dir: t.TempDir()}
	if w := benchmarkCommand(p, "status", "one", "127.0.0.1:1234"); w.Code != 200 {
		t.Fatal("serving PID discovery failed", w.Body.String())
	}
	defer func() {
		if p.file != nil {
			pprof.StopCPUProfile()
			p.file.Close()
		}
	}()
	for _, item := range []struct {
		action, name, address string
		code                  int
	}{
		{"start", "one", "10.0.0.1:1234", 403},
		{"start", "../escape", "127.0.0.1:1234", 400},
		{"invalid", "one", "127.0.0.1:1234", 400},
		{"stop", "one", "127.0.0.1:1234", 409},
	} {
		if got := benchmarkCommand(p, item.action, item.name, item.address).Code; got != item.code {
			t.Fatalf("%+v: got %d", item, got)
		}
	}
	w := benchmarkCommand(p, "start", "one", "127.0.0.1:1234")
	if w.Code != 200 {
		t.Fatal(w.Body.String())
	}
	if benchmarkCommand(p, "start", "two", "127.0.0.1:1234").Code != 409 ||
		benchmarkCommand(p, "stop", "two", "127.0.0.1:1234").Code != 409 {
		t.Fatal("overlap or stale stop accepted")
	}
	deadline := time.Now().Add(30 * time.Millisecond)
	for time.Now().Before(deadline) {
		_ = time.Now().UnixNano()
	}
	w = benchmarkCommand(p, "stop", "one", "127.0.0.1:1234")
	var result struct {
		PID     int    `json:"pid"`
		Path    string `json:"path"`
		Started int64  `json:"started_unix_ns"`
		Ended   int64  `json:"ended_unix_ns"`
	}
	if w.Code != 200 || json.Unmarshal(w.Body.Bytes(), &result) != nil || result.PID != os.Getpid() || result.Ended <= result.Started {
		t.Fatalf("invalid stop: %s", w.Body.String())
	}
	f, err := os.Open(result.Path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	z, err := gzip.NewReader(f)
	if err != nil {
		t.Fatal(err)
	}
	data, err := io.ReadAll(z)
	if err != nil || len(data) == 0 {
		t.Fatalf("incomplete profile: %v", err)
	}
	z.Close()
	if benchmarkCommand(p, "start", "one", "127.0.0.1:1234").Code != 409 {
		t.Fatal("existing profile was overwritten")
	}
}
