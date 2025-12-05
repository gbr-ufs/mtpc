DOCUMENT = main
OUT_DIR = ../dist

all: $(OUT_DIR) $(DOCUMENT).pdf

$(OUT_DIR):
	mkdir -p $@

$(DOCUMENT).pdf: $(DOCUMENT).tex
	latexmk -pdf -auxdir=build -outdir=$(OUT_DIR) $<

# Updates the file continuously.
watch: $(DOCUMENT).tex
	latexmk -pdf -pvc -auxdir=build -outdir=$(OUT_DIR) $<

clean:
	rm -rf build
