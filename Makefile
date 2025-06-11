# Makefile for building the site

PANDOC = pandoc
TEMPLATE = template.html
SOURCE = page.md
OUTPUT = index.html
CSS = style.css
WATCH_FILES = $(SOURCE) $(TEMPLATE) $(CSS) favicon.png
FAVICON_SIZES = favicon-16x16.png favicon-32x32.png apple-touch-icon.png apple-touch-icon-152x152.png apple-touch-icon-120x120.png mstile-144x144.png
FAVICON_SOURCE = favicon.png

# Default target: build the HTML and favicons
all: $(OUTPUT) favicons site.webmanifest

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

# Generate all favicon sizes
favicons: $(FAVICON_SIZES)

# Generate each favicon size
favicon-16x16.png: $(FAVICON_SOURCE)
	convert $(FAVICON_SOURCE) -resize 16x16 $@

favicon-32x32.png: $(FAVICON_SOURCE)
	convert $(FAVICON_SOURCE) -resize 32x32 $@

apple-touch-icon.png: $(FAVICON_SOURCE)
	convert $(FAVICON_SOURCE) -resize 180x180 $@

apple-touch-icon-152x152.png: $(FAVICON_SOURCE)
	convert $(FAVICON_SOURCE) -resize 152x152 $@

apple-touch-icon-120x120.png: $(FAVICON_SOURCE)
	convert $(FAVICON_SOURCE) -resize 120x120 $@

mstile-144x144.png: $(FAVICON_SOURCE)
	convert $(FAVICON_SOURCE) -resize 144x144 $@

# Generate site.webmanifest
site.webmanifest:
	@echo '{"name":"","short_name":"","icons":[{"src":"apple-touch-icon.png","sizes":"180x180","type":"image/png"}],"theme_color":"#ffffff","background_color":"#ffffff","display":"standalone"}' > $@

# Watch files (md, template, css, favicon) and rebuild on change
watch:
	@printf "%s\n" $(WATCH_FILES) | entr -r make

# Serve the site locally and watch for changes with LiveReload
dev: $(OUTPUT).livereload favicons site.webmanifest
	@echo "Starting server with LiveReload at http://localhost:8000..."
	livereload -p 8000 .

# Clean up generated files
clean:
	rm -f $(OUTPUT) $(OUTPUT).livereload $(FAVICON_SIZES) site.webmanifest

.PHONY: all watch dev clean $(OUTPUT).livereload favicons