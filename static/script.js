const form = document.getElementById("predictionForm");
const resultPanel = document.getElementById("result");
const statusHeader = document.getElementById("statusHeader");
const probabilityText = document.getElementById("probabilityText");
const decisionMetric = document.getElementById("decisionMetric");
const approvalMetric = document.getElementById("approvalMetric");
const riskMetric = document.getElementById("riskMetric");

const fields = [
    "dependents",
    "education",
    "self_employed",
    "income",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets",
    "commercial_assets",
    "luxury_assets",
    "bank_assets",
];

const samples = {
    approved: {
        dependents: 0,
        education: 0,
        self_employed: 0,
        income: 8000000,
        loan_amount: 1000000,
        loan_term: 5,
        cibil_score: 780,
        residential_assets: 3000000,
        commercial_assets: 2000000,
        luxury_assets: 1500000,
        bank_assets: 500000,
    },
    rejected: {
        dependents: 4,
        education: 1,
        self_employed: 1,
        income: 200000,
        loan_amount: 8000000,
        loan_term: 20,
        cibil_score: 350,
        residential_assets: 0,
        commercial_assets: 0,
        luxury_assets: 0,
        bank_assets: 5000,
    },
};

function fillSample(sample) {
    Object.entries(sample).forEach(([key, value]) => {
        document.getElementById(key).value = value;
    });
}

function buildPayload() {
    return Object.fromEntries(fields.map((field) => [field, document.getElementById(field).value]));
}

function setResult(state, title, message, approval = "--", risk = "--") {
    resultPanel.className = `result-panel ${state}`;
    statusHeader.textContent = title;
    probabilityText.textContent = message;
    decisionMetric.textContent = title;
    approvalMetric.textContent = approval;
    riskMetric.textContent = risk;
}

document.getElementById("sampleApproved").addEventListener("click", () => fillSample(samples.approved));
document.getElementById("sampleRejected").addEventListener("click", () => fillSample(samples.rejected));

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    setResult("", "Processing", "Evaluating the submitted application...");

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(buildPayload()),
        });

        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.error || "Prediction failed.");
        }

        const approval = `${result.approval_probability.toFixed(2)}%`;
        const risk = `${result.rejection_probability.toFixed(2)}%`;
        const state = result.approved ? "success" : "danger";
        const title = `Application ${result.decision}`;
        const message = `Approval confidence ${approval}. Rejection risk ${risk}.`;

        setResult(state, title, message, approval, risk);
    } catch (error) {
        setResult("danger", "Prediction Error", error.message);
    }
});
