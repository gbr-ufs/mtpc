DOCUMENT = main

all: $(DOCUMENT).pdf

$(DOCUMENT).pdf: $(DOCUMENT).tex

.PHONY: $(DOCUMENT).pdf

$(DOCUMENT).tex:
	latexmk -pdf -use-make -auxdir=build $@

.PHONY: $(DOCUMENT).tex

clean:
	latexmk -CA -auxdir=build
