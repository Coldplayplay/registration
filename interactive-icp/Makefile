BUILD_DIR = ./build
DEST_DIR = ./bin
DEST_EXE_NAME = tutorial_icp_demo

all: build
	cd $(BUILD_DIR);  make -j4


run:
	cd $(DEST_DIR); ./$(DEST_EXE_NAME)

clean:
	rm $(BUILD_DIR) -rf


build:
	mkdir $(BUILD_DIR);cd $(BUILD_DIR); cmake ..
