DOCUMENT = main
OUT_DIR = ../dist

all: $(DOCUMENT).pdf

$(DOCUMENT).pdf: $(DOCUMENT).tex
	latexmk -pdf -use-make -auxdir=build -outdir=$(OUT_DIR) $<

clean:
	latexmk -CA -auxdir=build
