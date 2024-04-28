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


@app.route('/ocr-parser', methods=['GET', 'POST'])
def ocr_parser():
    return render_template('ocr-app.html')

@app.route('/ocr-query', methods=['POST'])
def ocr_query():
    """
    Endpoint to process handwritten queries extracted from uploaded images.
    Input: Form data containing 'image_data' (file, base64-encoded image or direct file upload).
    Output: JSON with 'products' (array of objects) and 'response' (string).
    """
    if request.method == 'POST':
        # Get the uploaded file from the form
        image_file = request.files.get('image_data')
        # Check if a file was uploaded
        if image_file:
            # Secure the filename (prevents malicious characters)
            filename = secure_filename(image_file.filename)

            # Construct the full path for saving the image
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image_file.save(image_path)

            # Process the image to extract text and find matching products
            text = read_text(image_path)  # Assuming your read_text function accepts the full path
            products = list(product_query(text))  # Empty array, to be populated with product data
            response = output_response(products)  # Empty string, to be filled with a natural language response
            return jsonify({"products": products, "response": response})


@app.route('/image-product', methods=['GET', 'POST'])
def image_product():
    return render_template('image-product.html')

@app.route('/image-product-search', methods=['POST'])
def image_product_search():
  """
  Endpoint to identify and suggest products from uploaded product images.
  Input: Form data containing 'product_image' (file, base64-encoded image or direct file upload).
  Output: JSON with 'products' (array of objects) and 'response' (string).
  """
  if request.method == 'POST':
    # Get the uploaded file from the form
    product_image_file = request.files.get('product_image')

    if product_image_file:
      # Secure the filename (prevents malicious characters)
      filename = secure_filename(product_image_file.filename)

      # Construct the full path for saving the image
      image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
      product_image_file.save(image_path)

      # Predicting the class of the image
      predicted_class = predict(model_path, image_path)  # Use the saved image path
    #   printing predicted_class
      print(predicted_class)
      # Process the product image to detect and match products
      products = list(product_query(predicted_class))  # Empty array, to be populated with product data
      response = output_response(products)  # Empty string, to be filled with a natural language response

      # Get additional product information (assuming these functions exist)
      price = get_price(products)
      stock_code = get_stock_code(products)
      country = get_country(products)

      return jsonify({"stock_code": stock_code, "products": products, "response": response, "price": price, "country": country})
    else:
      return jsonify({"error": "No image uploaded"})

  # Handle case where request is not POST (optional)
  return jsonify({"error": "Invalid request method"})

@app.route('/sample_response', methods=['GET'])
def sample_response():
    """
    Endpoint to return a sample JSON response for the API.
    Output: JSON with 'products' (array of objects) and 'response' (string).
    """
    return render_template('sample_response.html')

if __name__ == '__main__':
    app.run(debug=True, port=8080)
