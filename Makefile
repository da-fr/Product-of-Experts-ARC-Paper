# Makefile for building the site

PANDOC = pandoc
TEMPLATE = template.html
SOURCE = page.md
OUTPUT = index.html
CSS = style.css
WATCH_FILES = $(SOURCE) $(TEMPLATE) $(CSS)

# Default target: build the HTML
all: $(OUTPUT)

# Build index.html from the Markdown source and template (without LiveReload)
$(OUTPUT): $(SOURCE) $(TEMPLATE)
	$(PANDOC) $(SOURCE) \
	  --from markdown \
	  --template=$(TEMPLATE) \
	  --output=$(OUTPUT) \
	  --variable=livereload:false

# Build with LiveReload enabled
$(OUTPUT).livereload: $(SOURCE) $(TEMPLATE)
	$(PANDOC) $(SOURCE) \
	  --from markdown \
	  --template=$(TEMPLATE) \
	  --output=$(OUTPUT) \
	  --variable=livereload:true
	@touch $(OUTPUT).livereload

# Watch files (md, template, css) and rebuild on change
watch:
	@printf "%s\n" $(WATCH_FILES) | entr -r make

# Serve the site locally and watch for changes with LiveReload
dev: $(OUTPUT).livereload
	@echo "Starting server with LiveReload at http://localhost:8000..."
	livereload -p 8000 .

# Clean up generated files
clean:
	rm -f $(OUTPUT) $(OUTPUT).livereload

.PHONY: all watch dev clean $(OUTPUT).livereload