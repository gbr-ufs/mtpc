DOCUMENT = main

all: $(DOCUMENT).pdf

$(DOCUMENT).pdf: $(DOCUMENT).tex

.PHONY: $(DOCUMENT).pdf

$(DOCUMENT).tex:
	cluttex -e pdflatex --biber=biber $@

.PHONY: $(DOCUMENT).tex
