# Task 1: Next Basket Prediction (LSTM)

### **Purpose**
Predicts the likelihood of a customer reordering items from their past purchase history. It acts as a **Quick Reorder Engine** for the marketplace.

### **Data Locations**
*   **Model Assets**: Root of `/task1_next_basket/` (`.keras` and `.pkl` files).
*   **Marketplace Data**: `/production_data/` (Exported from DESD).
*   **Research Data**: `/instacart_data/` (Used for Demo Mode parity tests).

### **Workflow**
1.  **Export**: Use the DESD Dashboard to "Export Next Basket Features".
2.  **Transfer**: Place the resulting CSVs into the `/production_data/` folder.
3.  **Inference**: The AAI service reads these CSVs to calculate real-time reorder probabilities for customers (e.g., Robert, ID 6).

### **Features Used**
The AI looks at:
- **User Stats**: Total orders and average days between shopping trips.
- **Product Stats**: Total marketplace popularity and global reorder rates.
- **User-Product Patterns**: How many times a specific user bought a specific item and how long it's been since their last purchase.