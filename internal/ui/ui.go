package ui

import (
	"fmt"
	"sync"
	"time"
)

// Spinner is an animated terminal spinner with a message.
type Spinner struct {
	message string
	stop    chan struct{}
	done    chan struct{}
	mu      sync.Mutex
}

// NewSpinner creates a new Spinner with the given initial message.
func NewSpinner(message string) *Spinner {
	return &Spinner{
		message: message,
		stop:    make(chan struct{}),
		done:    make(chan struct{}),
	}
}

// Start begins rendering the spinner in a background goroutine.
func (s *Spinner) Start() {
	go func() {
		defer close(s.done)
		frames := []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}
		i := 0
		for {
			select {
			case <-s.stop:
				// Clear the spinner line
				fmt.Printf("\r\033[K")
				return
			default:
				s.mu.Lock()
				msg := s.message
				s.mu.Unlock()
				fmt.Printf("\r\033[K  %s %s", frames[i%len(frames)], msg)
				i++
				time.Sleep(80 * time.Millisecond)
			}
		}
	}()
}

// Stop signals the spinner to stop and waits for it to clear the line.
func (s *Spinner) Stop() {
	close(s.stop)
	<-s.done
}

// UpdateMessage changes the spinner's displayed message while it's running.
func (s *Spinner) UpdateMessage(message string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.message = message
}
