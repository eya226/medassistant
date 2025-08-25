document.addEventListener('DOMContentLoaded', () => {
    // --- Element Selection ---
    const screens = {
        landing: document.getElementById('landing-screen'),
        upload: document.getElementById('upload-screen'),
        processing: document.getElementById('processing-screen'),
        results: document.getElementById('results-screen'),
    };

    const analyzeButton = document.getElementById('analyze-button');
    const analyzeAnotherButton = document.getElementById('analyze-another-button');

    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const fileInfo = document.getElementById('file-info');
    const uploadError = document.getElementById('upload-error');
    const processingText = document.getElementById('processing-text');
    const resultsContainer = document.getElementById('results-container');

    // --- Event Listeners ---
    analyzeButton.addEventListener('click', () => showScreen('upload'));
    analyzeAnotherButton.addEventListener('click', () => {
        resetUploadUI();
        showScreen('upload');
    });

    dropZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => handleFiles(e.target.files));
    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('hover'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('hover'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('hover');
        handleFiles(e.dataTransfer.files);
    });

    // --- File Handling and API Call ---
    function handleFiles(files) {
        if (files.length === 0) {
            showError('No files selected.');
            return;
        }

        fileInfo.textContent = `${files.length} file(s) selected.`;
        uploadError.style.display = 'none';

        showScreen('processing');
        processingText.textContent = `Analyzing ${files.length} scan(s)...`;
        uploadFiles(files);
    }

    async function uploadFiles(files) {
        const formData = new FormData();
        for (const file of files) {
            formData.append('files[]', file);
        }

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Server error during analysis');
            }

            const data = await response.json();
            displayResults(data);

        } catch (error) {
            showScreen('upload');
            showError(`Analysis failed: ${error.message}`);
        }
    }

    // --- UI Update Functions ---
    function displayResults(results) {
        // Clear previous results
        resultsContainer.innerHTML = '';

        // Define Triage Severity Order (higher number is more urgent)
        const severityOrder = {
            'Pneumothorax': 5,
            'Tuberculosis': 4,
            'Pneumonia': 4,
            'Edema': 3,
            'Atelectasis': 2,
            'Normal': 1,
            'Unknown': 0
        };

        // Sort results based on severity
        results.sort((a, b) => (severityOrder[b.prediction] || 0) - (severityOrder[a.prediction] || 0));

        // Create and append a result card for each result
        results.forEach(result => {
            const card = document.createElement('div');
            card.className = `result-card ${result.prediction}`;

            let innerHTML = '';
            if (result.error) {
                innerHTML = `<h3>${result.filename}</h3><p class="error-message" style="display:block;">Error: ${result.error}</p>`;
            } else {
                innerHTML = `
                    <div class="image-panel">
                        <img src="${result.image_url}" alt="Analyzed CT Scan">
                    </div>
                    <div class="results-panel">
                        <h3>${result.prediction}</h3>
                        <p class="confidence-score">File: ${result.filename} | Confidence: ${result.confidence}</p>
                        <p class="analysis-text">${result.analysis_text}</p>
                    </div>
                `;
            }
            card.innerHTML = innerHTML;
            resultsContainer.appendChild(card);
        });

        showScreen('results');
    }

    function showError(message) {
        uploadError.textContent = message;
        uploadError.style.display = 'block';
    }

    function resetUploadUI() {
        fileInfo.textContent = '';
        uploadError.style.display = 'none';
        fileInput.value = '';
    }

    function showScreen(screenName) {
        for (const screen in screens) {
            screens[screen].classList.remove('active');
        }
        screens[screenName].classList.add('active');
    }

    // Initial State
    showScreen('landing');
});
