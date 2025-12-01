SOURCE_DIR = src
OUT_DIR = dist

all:
	mkdir -p $(OUT_DIR)
	make -C $(SOURCE_DIR)

clean:
	rm -rf $(OUT_DIR)
