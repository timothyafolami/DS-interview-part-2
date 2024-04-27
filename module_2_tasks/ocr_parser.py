import easyocr

# Reader set up
reader = easyocr.Reader(['en'], detector='DB', recognizer = 'Transformer') 

# a function that takes an image and returns the text in the image
def read_text(image_path):
    result = reader.readtext(image_path)
    text = ''
    for i in result:
        text += i[1] + ' '
    return text