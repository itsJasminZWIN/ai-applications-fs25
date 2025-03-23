import joblib
import numpy as np
import gradio as gr

# Load your trained model
model = joblib.load("apartment_price_model.pkl")

# Define the Gradio prediction function
def predict_price(rooms, area, pop, pop_dens, frg_pct, emp, tax_income,
                  room_per_m2, luxurious, temporary, furnished, area_cat_ecoded, zurich_city):
    
    features = [[rooms, area, pop, pop_dens, frg_pct, emp, tax_income,
                 room_per_m2, luxurious, temporary, furnished, area_cat_ecoded, zurich_city]]

    prediction = model.predict(features)
    return round(prediction[0], 2)

# Create Gradio Interface
iface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Rooms"),
        gr.Number(label="Area (m²)"),
        gr.Number(label="Population"),
        gr.Number(label="Population Density"),
        gr.Number(label="Foreigner Percentage"),
        gr.Number(label="Employment Rate"),
        gr.Number(label="Tax Income"),
        gr.Number(label="Rooms per m²"),
        gr.Checkbox(label="Luxurious"),
        gr.Checkbox(label="Temporary"),
        gr.Checkbox(label="Furnished"),
        gr.Number(label="Area Category (Encoded)"),
        gr.Checkbox(label="Zurich City")
    ],
    outputs=gr.Number(label="Predicted Apartment Price (CHF)"),
    title="Apartment Price Prediction with Random Forest",
    examples=[
        [3, 80, 10000, 3000, 0.2, 5000, 100000, 1, 0, 0, 1, 1, 0],
        [4, 100, 20000, 5000, 0.3, 10000, 150000, 1, 1, 0, 0, 0, 1]
    ]
)

iface.launch()