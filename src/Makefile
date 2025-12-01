DOCUMENT = main
OUT_DIR = ../dist

all: $(DOCUMENT).pdf

$(DOCUMENT).pdf: $(DOCUMENT).tex
	cluttex -e pdflatex --biber=biber --output=$(OUT_DIR)/$@ $<
