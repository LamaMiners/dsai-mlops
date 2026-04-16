function sigmoid(z) {
    return 1 / (1 + Math.exp(-z));
}

async function predict() {
    try {
        const response = await fetch('model.json');
        const model = await response.json();
        const resDiv = document.getElementById('result');

        const features = [
            'num_words', 'num_unique_words', 'num_stopwords', 'num_links',
            'num_unique_domains', 'num_email_addresses', 'num_spelling_errors', 'num_urgent_keywords'
        ];

        let z = model.intercept;
        features.forEach((f, index) => {
            const val = parseFloat(document.getElementById(f).value) || 0;
            z += val * model.coef[index];
        });

        const prob = sigmoid(z);
        const isPhishing = prob >= 0.5;

        resDiv.innerHTML = isPhishing 
            ? `<span class="text-danger">⚠️ PHISHING (${(prob*100).toFixed(1)}%)</span>`
            : `<span class="text-success">✅ SAFE (${((1-prob)*100).toFixed(1)}%)</span>`;
            
    } catch (e) {
        console.error(e);
        document.getElementById('result').innerText = "Error loading model!";
    }
}