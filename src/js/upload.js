document.addEventListener('DOMContentLoaded', () => {
    const STYLE_DATA = {
        "Achaemenid architecture": "Grand monumental architecture with massive columns.",
        "American Foursquare architecture": "Simple, boxy design with a hipped roof and wide porch.",
        "American craftsman style": "Handcrafted details, low-pitched roofs, and exposed rafters.",
        "Ancient Egyptian architecture": "Massive scale, pyramids, and temples with thick walls.",
        "Art Deco architecture": "Sleek geometric forms, bold lines, and glamorous ornamentation.",
        "Art Nouveau architecture": "Flowing, organic lines inspired by nature and floral motifs.",
        "Baroque architecture": "Dynamic forms, grand scale, and elaborate ornamentation.",
        "Bauhaus architecture": "Functional, minimalist design unifying art, craft, and technology.",
        "Beaux-Arts architecture": "Grand, theatrical, and highly decorative neoclassical style.",
        "Byzantine architecture": "Massive domes, round arches, and rich mosaic decorations.",
        "Chicago school architecture": "Early steel-frame skyscrapers with large plate-glass windows.",
        "Colonial architecture": "Symmetrical, classical details based on early settlements.",
        "Deconstructivism": "Fragmented, non-linear, and visually unpredictable forms.",
        "Edwardian architecture": "Lighter, less ornate than Victorian, with classical influences.",
        "Georgian architecture": "Strict symmetry, classical proportions, and brick facades.",
        "Gothic architecture": "Pointed arches, ribbed vaults, and flying buttresses.",
        "Greek Revival architecture": "Temple-like fronts, sturdy columns, and classical symmetry.",
        "International style": "Volume over mass, flat surfaces, and rejection of ornament.",
        "Novelty architecture": "Playful, literal shapes designed as advertisements or attractions.",
        "Palladian architecture": "Classical antiquity revival, strict symmetry, and Venetian windows.",
        "Postmodern architecture": "Playful mix of historical references, irony, and bright colors.",
        "Queen Anne architecture": "Asymmetrical facades, wrap-around porches, and elaborate trims.",
        "Romanesque architecture": "Thick walls, round arches, sturdy pillars, and large towers.",
        "Tudor Revival architecture": "Half-timbering, steep gable roofs, and tall mullioned windows."
    };
    
    const CLASS_NAMES = Object.keys(STYLE_DATA);

    const uploadInput = document.getElementById('imageInput');
    const uploadSection = document.querySelector('.upload-section');
    const resultsSection = document.getElementById('results-section');
    const uploadedImg = document.getElementById('uploaded-image');
    const filenameBadge = document.getElementById('filename-badge');
    const resetBtn = document.getElementById('reset-btn');
    
    const scannerContainer = document.getElementById('scanner-container');
    const predictionsWrapper = document.getElementById('predictions-wrapper');
    const allStylesWrapper = document.getElementById('all-styles-wrapper');

    const toggleDetailsBtn = document.getElementById('toggle-details-btn');
    const stylesList = document.getElementById('styles-list');
    const hintText = document.getElementById('detailed-hint-text');

    toggleDetailsBtn.addEventListener('click', () => {
        stylesList.classList.toggle('hidden');
        hintText.innerText = stylesList.classList.contains('hidden') ? '↓ click to expand' : '↑ click to collapse';
    });

    resetBtn.addEventListener('click', () => {
        resultsSection.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        uploadInput.value = '';
    });

    const uploadContainer = document.getElementById('uploadContainer');
    if (uploadContainer) {
        uploadContainer.addEventListener('click', () => uploadInput.click());
    }

    let session = null;

    async function loadModel() {
        try {
            session = await ort.InferenceSession.create('src/models/arch_vision_full.onnx');
            console.log("Model loaded successfully!");
        } catch (e) {
            console.error("Model load error:", e);
        }
    }
    loadModel();

    uploadInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (!file) return;

        filenameBadge.innerText = file.name;

        const reader = new FileReader();
        reader.onload = function(event) {
            uploadedImg.src = event.target.result;
            
            uploadSection.classList.add('hidden');
            resultsSection.classList.remove('hidden');
            
            scannerContainer.style.display = 'flex';
            predictionsWrapper.classList.add('hidden');
            allStylesWrapper.classList.add('hidden');
            stylesList.classList.add('hidden');
            hintText.innerText = '↓ click to expand';

            uploadedImg.onload = () => {
                runInference(uploadedImg);
            };
        };
        reader.readAsDataURL(file);
    });

    function preprocessImage(image) {
        const targetEdge = 256;
        const cropSize = 224;

        let width = image.width;
        let height = image.height;
        if (width < height) {
            height = (height / width) * targetEdge;
            width = targetEdge;
        } else {
            width = (width / height) * targetEdge;
            height = targetEdge;
        }

        const canvas = document.createElement('canvas');
        canvas.width = cropSize;
        canvas.height = cropSize;
        const ctx = canvas.getContext('2d');

        const dx = (cropSize - width) / 2;
        const dy = (cropSize - height) / 2;
        ctx.drawImage(image, dx, dy, width, height);

        const imageData = ctx.getImageData(0, 0, cropSize, cropSize).data;
        const float32Data = new Float32Array(3 * cropSize * cropSize);
        const mean = [0.485, 0.456, 0.406];
        const std = [0.229, 0.224, 0.225];

        for (let i = 0; i < cropSize * cropSize; i++) {
            let r = imageData[i * 4] / 255.0;
            let g = imageData[i * 4 + 1] / 255.0;
            let b = imageData[i * 4 + 2] / 255.0;

            float32Data[i] = (r - mean[0]) / std[0]; 
            float32Data[cropSize * cropSize + i] = (g - mean[1]) / std[1];
            float32Data[2 * cropSize * cropSize + i] = (b - mean[2]) / std[2];
        }
        return new ort.Tensor('float32', float32Data, [1, 3, cropSize, cropSize]);
    }

    function softmax(arr) {
        const max = Math.max(...arr);
        const exps = arr.map(x => Math.exp(x - max));
        const sumExps = exps.reduce((a, b) => a + b, 0);
        return exps.map(x => (x / sumExps) * 100);
    }

    async function runInference(imageElement) {
        if (!session) {
            alert("Model is still loading, please wait a moment.");
            return;
        }

        try {
            const tensor = preprocessImage(imageElement);
            const feeds = { 'input': tensor };
            const results = await session.run(feeds);
            const outputLogits = results.output.data; 
            const probabilities = softmax(Array.from(outputLogits));

            let predictions = probabilities.map((prob, index) => ({
                name: CLASS_NAMES[index],
                desc: STYLE_DATA[CLASS_NAMES[index]],
                prob: prob
            })).sort((a, b) => b.prob - a.prob);

            displayResults(predictions);

        } catch (e) {
            console.error("Inference error:", e);
            alert("An error occurred during analysis.");
        }
    }

    function displayResults(predictions) {
        scannerContainer.style.display = 'none';
        predictionsWrapper.classList.remove('hidden');
        allStylesWrapper.classList.remove('hidden');

        const formatName = (name) => name.replace(" architecture", "");

        document.getElementById('top-style-name').innerText = formatName(predictions[0].name);
        document.getElementById('top-style-desc').innerText = predictions[0].desc;
        document.getElementById('top-style-percent').innerText = predictions[0].prob.toFixed(1) + '%';

        setTimeout(() => {
            document.getElementById('top-style-progress').style.width = predictions[0].prob + '%';
        }, 100);

        document.getElementById('match-2-name').innerText = formatName(predictions[1].name);
        document.getElementById('match-2-pct').innerText = predictions[1].prob.toFixed(0) + '%';
        
        document.getElementById('match-3-name').innerText = formatName(predictions[2].name);
        document.getElementById('match-3-pct').innerText = predictions[2].prob.toFixed(0) + '%';

        document.getElementById('match-4-name').innerText = formatName(predictions[3].name);
        document.getElementById('match-4-pct').innerText = predictions[3].prob.toFixed(0) + '%';

        stylesList.innerHTML = '';
        predictions.forEach(item => {
            const row = document.createElement('div');
            row.className = 'style-row';
            row.innerHTML = `
                <div class="style-name-col">${formatName(item.name)}</div>
                <div class="style-bar-bg">
                    <div class="style-bar-fill" style="width: 0%"></div>
                </div>
                <div class="style-percent-col">${item.prob.toFixed(1)}%</div>
            `;
            stylesList.appendChild(row);

            setTimeout(() => {
                row.querySelector('.style-bar-fill').style.width = item.prob + '%';
            }, 50);
        });
    }
});