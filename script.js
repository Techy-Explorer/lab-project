function sendExpense() {

  const data = {
    date: document.getElementById("date").value,
    mode: document.getElementById("mode").value,
    category: document.getElementById("category").value,
    subcategory: document.getElementById("subcategory").value,
    note: document.getElementById("note").value,
    amount: parseFloat(document.getElementById("amount").value),
    type: document.getElementById("type").value,
    currency: document.getElementById("currency").value,
    anomaly: parseInt(document.getElementById("anomaly").value),
    day: parseInt(document.getElementById("day").value),
    month: parseInt(document.getElementById("month").value),
    year: parseInt(document.getElementById("year").value)
  };

  fetch("http://127.0.0.1:8000/predict-expense-full", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data)
  })
    .then(res => res.json())
    .then(res => alert("Predicted Expense: " + res.predicted_expense))
    .catch(err => alert("Error: " + err));
}
