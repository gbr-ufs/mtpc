DOCUMENT = main

PHONY: $(DOCUMENT).pdf

all: $(DOCUMENT).pdf

$(DOCUMENT).pdf:
	latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make $(DOCUMENT).tex

clean:
	latexmk -CA
