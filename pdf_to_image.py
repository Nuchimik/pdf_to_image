#!/usr/bin/env python
import os
import sys
import logging
from datetime import datetime
import numpy as np
import cv2
from PyPDF2 import PdfFileReader


def get_pages_pdf(filepath):
    """
    Reading pdf file and getting all pages
    """
    document = PdfFileReader(filepath, "rb")
    pagesCount = document.trailer['/Root']['/Pages']['/Count']
    pages = [document.getPage(page) for page in range(pagesCount)]
    return pages


def page_to_array(page):
    """
    Reading the page-stream of pdf
    """
    xObject = page['/Resources']['/XObject'].getObject()
    if not xObject:
        return None
    pageObject = None
    for obj in xObject:
        if xObject[obj]['/Subtype'] == '/Image':
            pageObject = xObject[obj]
            break
    if not pageObject:
        return pageObject
    imageArray = np.fromstring(pageObject._data, dtype=np.uint8)
    return imageArray


def decode_array(imageArray):
    """
    Reads an image from a buffer in memory
    """
    image = cv2.imdecode(imageArray, cv2.IMREAD_UNCHANGED)
    return image


def rotate_image(image):
    """
    Automatic alignment of content orientation
    """
    tmpImage = image.copy()
    tmpImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tmpImage = cv2.bitwise_not(tmpImage)
    tmpImage = cv2.threshold(tmpImage, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(tmpImage > 0))
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width//2, height//2), angle, 1.0)
    rotatedImage = cv2.warpAffine(
        image, matrix, (width, height),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotatedImage


def resize_image(image, scale=300, interpolation=cv2.INTER_CUBIC):
    """
    Image resizing.
    Tessarakt gives the best results in high-resolution images
    """
    width = int(image.shape[1] * scale / 100) 
    height = int(image.shape[0] * scale / 100) 
    resizedImage = cv2.resize(image, (width, height), interpolation=interpolation)
    return resizedImage


def crop_image(image):
    """
    Cropping blank edges in the image
    """
    tmpImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blackPoints = np.argwhere(tmpImage==0)
    blackPoints = np.fliplr(blackPoints)
    xStart, yStart, wBox, hBox = cv2.boundingRect(blackPoints)
    xMin, xMax, yMin, yMax = xStart, xStart + wBox, yStart, yStart + hBox
    croppedImage = image[yMin:yMax, xMin:xMax]
    return croppedImage


def convert_page(page):
    """
    Converting image from PDF
    """
    array = page_to_array(page)
    image = decode_array(array)
    return image


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")

    if not os.path.isdir(LOGS_DIR):
        os.mkdir(LOGS_DIR)

    if not os.path.isdir(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)

    LOGFILE = os.path.join(LOGS_DIR, f"Logs for {datetime.now():%H.%M %Y-%m-%d}.log")
    logging.basicConfig(
        handlers=[logging.FileHandler(LOGFILE, "w", 'utf-8')],
        level=logging.INFO,
        format='%(message)s'
    )

    filename = "example.pdf"
    filepath = os.path.join(BASE_DIR, filename)

    # Get pages from pdf
    pages = get_pages_pdf(filename)

    # Test image
    page = pages[0]

    # Page to np-array
    page = convert_page(page)
    print(type(page))

    # Convert image for cv2 processing
    image1 = rotate_image(page)
    cv2.imwrite(os.path.join(RESULTS_DIR, "rotate_image.png"), image1)

    # Resize image
    image2 = resize_image(page)
    cv2.imwrite(os.path.join(RESULTS_DIR, "resize_image.png"), image2)

    # Convert image for cv2 processing
    image3 = crop_image(image1)
    cv2.imwrite(os.path.join(RESULTS_DIR, "crop_image.png"), image3)


if __name__ == '__main__':
    main()
