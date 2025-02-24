#!/bin/bash


if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <source directory> <output directory>"
  exit 1
fi

SOURCE_DIR=$1
OUTPUT_DIR=$2

# Ensure directories have a trailing slash
SOURCE_DIR="${SOURCE_DIR%/}/"
OUTPUT_DIR="${OUTPUT_DIR%/}/"

# Run GIMP batch script
gimp -n -i -b - <<EOF
(let* ( 
    (source-dir "$SOURCE_DIR")  
    (output-dir "$OUTPUT_DIR")  
    (file's (cadr (file-glob (string-append source-dir "*.xcf") 1))) 
    (filename "") 
    (image 0) 
    (layer 0) 
    (mode 0)
  )
  (while (pair? file's)
    (set! image (car (gimp-file-load RUN-NONINTERACTIVE (car file's) (car file's))))
    (set! mode (car (gimp-image-base-type image)))

    ;; Convert all images to grayscale
    (cond
      ((= mode RGB) (gimp-image-convert-grayscale image))
      ((= mode INDEXED) (gimp-image-convert-grayscale image))
    )

    ;; (set! layer (car (gimp-image-merge-visible-layers image CLIP-TO-IMAGE)))
    ;; Remove alpha channel (ensure 1-channel grayscale output)
    (set! layer (car (gimp-image-flatten image)))

    (set! filename (string-append output-dir 
                                  (substring (car file's) 
                                             (+ (string-length source-dir) 0) 
                                             (- (string-length (car file's)) 4)) 
                                  ".png"))
    (gimp-file-save RUN-NONINTERACTIVE image layer filename filename)
    (gimp-image-delete image)
    (set! file's (cdr file's))
    )
  (gimp-quit 0)
  )
EOF
#gimp -n -i -b - <<EOF
#(let* (
#    (source-dir "$SOURCE_DIR")
#    (output-dir "$OUTPUT_DIR")
#    (file's (cadr (file-glob (string-append source-dir "*.xcf") 1)))
#    (filename "")
#    (image 0)
#    (layer 0)
#  )
#  (while (pair? file's)
#    (set! image (car (gimp-file-load RUN-NONINTERACTIVE (car file's) (car file's))))
#    (gimp-image-convert-grayscale image)
#    (set! layer (car (gimp-image-merge-visible-layers image CLIP-TO-IMAGE)))
#    (set! filename (string-append output-dir 
#                                  (substring (car file's) 
#                                             (+ (string-length source-dir) 0)
#                                             (- (string-length (car file's)) 4))
#                                  ".png"))
#    (gimp-file-save RUN-NONINTERACTIVE image layer filename filename)
#    (gimp-image-delete image)
#    (set! file's (cdr file's))
#    )
#  (gimp-quit 0)
#  )
#EOF


#gimp -n -i -b - <<EOF
#(let* ( (file's (cadr (file-glob "*.xcf" 1))) (filename "") (image 0) (layer 0) )
#  (while (pair? file's)
#    (set! image (car (gimp-file-load RUN-NONINTERACTIVE (car file's) (car file's))))
#    (set! layer (car (gimp-image-merge-visible-layers image CLIP-TO-IMAGE)))
#    (set! filename (string-append (substring (car file's) 0 (- (string-length (car file's)) 4)) ".png"))
#    (gimp-file-save RUN-NONINTERACTIVE image layer filename filename)
#    (gimp-image-delete image)
#    (set! file's (cdr file's))
#    )
#  (gimp-quit 0)
#  )
#EOF
