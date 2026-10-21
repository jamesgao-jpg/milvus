// Benchmark CPU control captures fixed request batches without changing Query execution.
package http

import (
	"encoding/json"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"runtime/pprof"
	"strconv"
	"sync"
	"time"
)

// benchmarkCPUProfile owns only the CPU capture opened by the benchmark runner.
type benchmarkCPUProfile struct {
	mu      sync.Mutex
	dir     string
	file    *os.File
	name    string
	started int64
}

// registerBenchmarkCPUProfile leaves the control route absent unless explicitly enabled.
func registerBenchmarkCPUProfile(dir string) {
	if dir != "" {
		control := &benchmarkCPUProfile{dir: dir}
		Register(&Handler{Path: "/debug/benchmark/cpu", HandlerFunc: control.serveHTTP})
	}
}

// serveHTTP acknowledges sampling boundaries; stopping flushes the complete profile.
func (p *benchmarkCPUProfile) serveHTTP(w http.ResponseWriter, r *http.Request) {
	// Restrict control to the local runner and reject malformed or stale commands.
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil || !net.ParseIP(host).IsLoopback() {
		http.Error(w, "benchmark CPU control is loopback-only", http.StatusForbidden)
		return
	}
	if r.Method != http.MethodPost {
		http.Error(w, "POST required", http.StatusMethodNotAllowed)
		return
	}
	name, action := r.URL.Query().Get("name"), r.URL.Query().Get("action")
	if !regexp.MustCompile(`^[a-zA-Z0-9_-]{1,160}$`).MatchString(name) || (action != "start" && action != "stop" && action != "status") {
		http.Error(w, "valid name and start/stop action required", http.StatusBadRequest)
		return
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	ended := int64(0)
	path := filepath.Join(p.dir, name+"-"+strconv.Itoa(os.Getpid())+".cpu.pprof")
	if action == "start" {
		// Exclusive files prevent overwriting earlier evidence or another active capture.
		if p.file != nil {
			http.Error(w, "benchmark CPU capture already active", http.StatusConflict)
			return
		}
		if err := os.MkdirAll(p.dir, 0o700); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
		if err != nil {
			http.Error(w, err.Error(), http.StatusConflict)
			return
		}
		if err := pprof.StartCPUProfile(file); err != nil {
			file.Close()
			os.Remove(path)
			http.Error(w, err.Error(), http.StatusConflict)
			return
		}
		p.file, p.name, p.started = file, name, time.Now().UnixNano()
	} else if action == "stop" {
		// A stop for an older interval must not terminate a newer capture.
		if p.file == nil || p.name != name {
			http.Error(w, "no matching benchmark CPU capture", http.StatusConflict)
			return
		}
		ended = time.Now().UnixNano()
		pprof.StopCPUProfile()
		err := p.file.Close()
		p.file = nil
		if err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"pid": os.Getpid(), "path": path, "started_unix_ns": p.started,
		"ended_unix_ns": ended, "acknowledged_unix_ns": time.Now().UnixNano(), "active": p.file != nil,
	})
}
