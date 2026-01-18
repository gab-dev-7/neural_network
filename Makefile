CC = gcc
CFLAGS = -Wall -Wextra -O2 -g -std=c99
LDFLAGS = -lm

TARGET = neural_network

SRCS = main.c neural_network.c
OBJS = $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS) $(LDFLAGS)

main.o: main.c neural_network.h
	$(CC) $(CFLAGS) -c main.c

neural_network.o: neural_network.c neural_network.h
	$(CC) $(CFLAGS) -c neural_network.c

run: $(TARGET)
	./$(TARGET) -v

test: $(TARGET)
	./$(TARGET) -e 5000 -l 0.3 -h 8 -v

debug: CFLAGS += -DDEBUG -ggdb3
debug: clean $(TARGET)

clean:
	rm -f $(TARGET) $(OBJS) *.out

install: $(TARGET)
	cp $(TARGET) /usr/local/bin/

uninstall:
	rm -f /usr/local/bin/$(TARGET)

dist: clean
	mkdir -p dist/neural_network
	cp *.c *.h Makefile README.md dist/neural_network/
	tar -czf neural_network.tar.gz -C dist .
	rm -rf dist

help:
	@echo "Available targets:"
	@echo "  all     - Build the program (default)"
	@echo "  run     - Build and run with verbose mode"
	@echo "  test    - Run with test parameters"
	@echo "  debug   - Build with debug symbols"
	@echo "  clean   - Remove build files"
	@echo "  install - Install to /usr/local/bin (requires sudo)"
	@echo "  uninstall - Uninstall from /usr/local/bin"
	@echo "  dist    - Create distribution archive"
	@echo "  help    - Show this help message"

.PHONY: all run test debug clean install uninstall dist help
