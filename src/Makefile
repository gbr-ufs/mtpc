DOCUMENT = main
OUT_DIR = ../dist

all: $(DOCUMENT).pdf

$(DOCUMENT).pdf: $(DOCUMENT).tex
	latexmk -pdf -auxdir=build -outdir=$(OUT_DIR) $<

# Updates the file continuously.
watch: $(DOCUMENT).tex
	latexmk -pdf -pvc -auxdir=build -outdir=$(OUT_DIR) $<

clean:
	rm -rf build
