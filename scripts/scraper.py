import requests
from bs4 import BeautifulSoup
import pandas as pd

# WEBSITE URL
url = "https://books.toscrape.com/"

# SEND REQUEST
response = requests.get(url)

# CHECK STATUS
print("Status Code:", response.status_code)

# PARSE HTML
soup = BeautifulSoup(response.text, "html.parser")

# FIND PRODUCTS
products = soup.find_all("article", class_="product_pod")

# EMPTY LIST
data = []

# LOOP THROUGH PRODUCTS
for product in products:
    
    # PRODUCT NAME
    name = product.h3.a["title"]
    
    # PRODUCT PRICE
    price = product.find("p", class_="price_color").text
    
    # APPEND DATA
    data.append([name, price])

# CREATE DATAFRAME
df = pd.DataFrame(data, columns=["Product", "Price"])

# DISPLAY FIRST 5 ROWS
print(df.head())

# SAVE CSV
df.to_csv("data/raw/products.csv", index=False)

print("CSV Saved Successfully!")