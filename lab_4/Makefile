CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11 -Ilibs
LDFLAGS = -lm

BUILD_DIR = build
SRC_DIR = src
LIBS = libs
TARGET = $(BUILD_DIR)/pathtrace

SCENE = assets/scenes/scene.json
TEST_OUT = renders/c_test.ppm

.PHONY: all clean test run

all: $(TARGET)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(TARGET): $(SRC_DIR)/main.c $(LIBS)/cJSON.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $(SRC_DIR)/main.c $(LIBS)/cJSON.c $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR)

test: $(TARGET)
	./$(TARGET) --scene $(SCENE) --w 64 --h 64 --spp 2 --max-depth 6 --out $(TEST_OUT)

run: $(TARGET)
	./$(TARGET) --scene $(SCENE) --out renders/output.ppm
