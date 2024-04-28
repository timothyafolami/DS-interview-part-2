from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
from module_1_tasks.product_search import product_query, output_response
from module_2_tasks.ocr_parser import read_text
from module_3_task.model_inference import predict
import os
from werkzeug.utils import secure_filename
from utils import get_most_recent_file, get_price, get_stock_code, get_country


UPLOAD_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_path = "./module_3_task/my_model_weights.pth"


@app.route('/product-rec', methods=['GET'])
def welcome():
    return render_template('product-rec.html')


@app.route('/product-recommendation', methods=['POST'])
def product_recommendation():
    """
    Endpoint for product recommendations based on natural language queries.
    Input: Form data containing 'query' (string).
    Output: JSON with 'products' (array of objects) and 'response' (string).
    """
    query = request.form.get('query', '')
    # Process the query to find matching products
    products = list(product_query(query))  # Empty array, to be populated with product data
    response = output_response(products)  # Empty string, to be filled with a natural language response
    return jsonify({"products": products, "response": response})


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/ocr-parser', methods=['GET', 'POST'])
def ocr_parser():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('download_file', name=filename))

    return render_template('ocr-app.html')

@app.route('/ocr-query', methods=['POST'])
def ocr_query():
    """
    Endpoint to process handwritten queries extracted from uploaded images.
    Input: Form data containing 'image_data' (file, base64-encoded image or direct file upload).
    Output: JSON with 'products' (array of objects) and 'response' (string).
    """
    image_file = str(get_most_recent_file())
    print(image_file)
    # Process the image to extract text and find matching products
    text = read_text(image_file)
    products = list(product_query(text))  # Empty array, to be populated with product data
    response = output_response(products)  # Empty string, to be filled with a natural language response
    return jsonify({"products": products, "response": response})


@app.route('/image-product', methods=['GET', 'POST'])
def image_product():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('download_file', name=filename))

    return render_template('image-product.html')

@app.route('/image-product-search', methods=['POST'])
def image_product_search():
    """
    Endpoint to identify and suggest products from uploaded product images.
    Input: Form data containing 'product_image' (file, base64-encoded image or direct file upload).
    Output: JSON with 'products' (array of objects) and 'response' (string).
    """
    product_image = str(get_most_recent_file())
    # predicting the class of the image
    predicted_class = predict(model_path, product_image)
    # Process the product image to detect and match products
    products = list(product_query(predicted_class))  # Empty array, to be populated with product data
    response = output_response(products)  # Empty string, to be filled with a natural language response
    # get price and description
    price = get_price(products)
    stock_code = get_stock_code(products)
    country = get_country(products)
    return jsonify({"stock_code": stock_code,"products": products, "response": response, "price": price, "country": country})

@app.route('/sample_response', methods=['GET'])
def sample_response():
    """
    Endpoint to return a sample JSON response for the API.
    Output: JSON with 'products' (array of objects) and 'response' (string).
    """
    return render_template('sample_response.html')

if __name__ == '__main__':
    app.run(debug=True, port=8080)
