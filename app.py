from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="GPU Performance Prediction API")

model = joblib.load("model.pkl")


class GPUInput(BaseModel):
    id: int
    cores: float
    tensor_cores: float
    rt_cores: float
    sm_count: float
    base_clock: float
    boost_clock: float
    memory_clock: float
    ram: float
    bus_width: float
    fp16_performance: float
    fp32_performance: float
    fp64_performance: float
    bf16_performance: float
    tf32_performance: float
    api_cuda: float


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>GPU Performance Predictor</title>

    <style>
        body {
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            margin: 0;
            padding: 30px;
        }

        .container {
            max-width: 900px;
            margin: auto;
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }

        h1 {
            text-align: center;
            margin-bottom: 10px;
        }

        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }

        label {
            font-weight: bold;
            display: block;
            margin-bottom: 5px;
        }

        input {
            width: 100%;
            padding: 10px;
            box-sizing: border-box;
            border: 1px solid #ccc;
            border-radius: 7px;
        }

        button {
            width: 100%;
            margin-top: 25px;
            padding: 14px;
            border: none;
            border-radius: 8px;
            background: #2563eb;
            color: white;
            font-size: 17px;
            cursor: pointer;
        }

        button:hover {
            background: #1d4ed8;
        }

        #result {
            margin-top: 25px;
            padding: 18px;
            text-align: center;
            background: #eef6ff;
            border-radius: 8px;
            font-size: 20px;
            font-weight: bold;
        }

        @media (max-width: 700px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>

<body>

<div class="container">

    <h1>GPU Performance Predictor</h1>

    <p class="subtitle">
        Enter GPU specifications to predict the GPI value
    </p>

    <div class="form-grid">

        <div>
            <label>ID</label>
            <input id="id" type="number" value="1">
        </div>

        <div>
            <label>Cores</label>
            <input id="cores" type="number" value="3584">
        </div>

        <div>
            <label>Tensor Cores</label>
            <input id="tensor_cores" type="number" value="112">
        </div>

        <div>
            <label>RT Cores</label>
            <input id="rt_cores" type="number" value="28">
        </div>

        <div>
            <label>SM Count</label>
            <input id="sm_count" type="number" value="28">
        </div>

        <div>
            <label>Base Clock</label>
            <input id="base_clock" type="number" value="1500">
        </div>

        <div>
            <label>Boost Clock</label>
            <input id="boost_clock" type="number" value="1800">
        </div>

        <div>
            <label>Memory Clock</label>
            <input id="memory_clock" type="number" value="7000">
        </div>

        <div>
            <label>RAM</label>
            <input id="ram" type="number" value="8">
        </div>

        <div>
            <label>Bus Width</label>
            <input id="bus_width" type="number" value="256">
        </div>

        <div>
            <label>FP16 Performance</label>
            <input id="fp16_performance" type="number" value="10000">
        </div>

        <div>
            <label>FP32 Performance</label>
            <input id="fp32_performance" type="number" value="5000">
        </div>

        <div>
            <label>FP64 Performance</label>
            <input id="fp64_performance" type="number" value="150">
        </div>

        <div>
            <label>BF16 Performance</label>
            <input id="bf16_performance" type="number" value="10000">
        </div>

        <div>
            <label>TF32 Performance</label>
            <input id="tf32_performance" type="number" value="5000">
        </div>

        <div>
            <label>API CUDA</label>
            <input id="api_cuda" type="number" value="12">
        </div>

    </div>

    <button onclick="predict()">Predict GPI</button>

    <div id="result">
        Enter GPU details and click Predict GPI
    </div>

</div>


<script>

async function predict() {

    const fields = [
        "id",
        "cores",
        "tensor_cores",
        "rt_cores",
        "sm_count",
        "base_clock",
        "boost_clock",
        "memory_clock",
        "ram",
        "bus_width",
        "fp16_performance",
        "fp32_performance",
        "fp64_performance",
        "bf16_performance",
        "tf32_performance",
        "api_cuda"
    ];

    const data = {};

    fields.forEach(field => {
        data[field] = Number(document.getElementById(field).value);
    });

    document.getElementById("result").innerHTML = "Predicting...";

    try {

        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || "Prediction failed");
        }

        document.getElementById("result").innerHTML =
            "Predicted GPI Value: " +
            Number(result.predicted_gpi_value).toFixed(4);

    } catch (error) {

        document.getElementById("result").innerHTML =
            "Error: " + error.message;
    }
}

</script>

</body>
</html>
"""


@app.post("/predict")
def predict(data: GPUInput):

    input_data = pd.DataFrame([data.model_dump()])

    input_data["avg_clock"] = (
        input_data["base_clock"] +
        input_data["boost_clock"]
    ) / 2

    input_data["clock_difference"] = (
        input_data["boost_clock"] -
        input_data["base_clock"]
    )

    input_data["memory_bandwidth"] = (
        input_data["memory_clock"] *
        input_data["bus_width"]
    )

    prediction = model.predict(input_data)[0]

    return {
        "predicted_gpi_value": float(prediction)
    }


if __name__ == "__main__":

    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )