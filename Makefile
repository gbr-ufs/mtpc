DOCUMENT = main
SOURCE_DIR = src
DIST_DIR = dist

all:
	make -C $(SOURCE_DIR)
	mkdir -p $(DIST_DIR)
	mv $(SOURCE_DIR)/$(DOCUMENT).pdf $(DIST_DIR)

clean:
	rm -rf $(DIST_DIR)
	make -C $(SOURCE_DIR) $@
