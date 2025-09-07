document.addEventListener('DOMContentLoaded', () => {
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

    const showScreen = (screenName) => {
        Object.values(screens).forEach(screen => screen.classList.remove('active'));
        screens[screenName].classList.add('active');
    };

    analyzeButton.addEventListener('click', () => showScreen('upload'));
    analyzeAnotherButton.addEventListener('click', () => {
        // Reset the UI and then immediately trigger the file input dialog
        fileInfo.textContent = '';
        uploadError.style.display = 'none';
        fileInput.value = ''; // Clear previous selection
        fileInput.click(); // Open file dialog for the user
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

    const handleFiles = (files) => {
        if (files.length === 0) return;
        fileInfo.textContent = `${files.length} file(s) selected.`;
        uploadError.style.display = 'none';
        showScreen('processing');
        processingText.textContent = `Analyzing ${files.length} scan(s)...`;
        uploadFiles(files);
    };

    const uploadFiles = async (files) => {
        const formData = new FormData();
        for (const file of files) {
            formData.append('files[]', file);
        }
        try {
            const response = await fetch('/predict', { method: 'POST', body: formData });
            if (!response.ok) throw new Error((await response.json()).error || 'Server error');
            displayResults(await response.json());
        } catch (error) {
            showScreen('upload');
            uploadError.textContent = `Analysis failed: ${error.message}`;
            uploadError.style.display = 'block';
        }
    };

    const displayResults = (results) => {
        resultsContainer.innerHTML = '';
        const severityOrder = { 'glioma': 4, 'meningioma': 3, 'pituitary': 2, 'notumor': 1, 'Unknown': 0 };
        results.sort((a, b) => (severityOrder[b.prediction] || 0) - (severityOrder[a.prediction] || 0));

        results.forEach(result => {
            const card = document.createElement('div');
            card.className = `result-card ${result.prediction}`;
            let innerHTML = result.error
                ? `<h3>${result.filename}</h3><p class="error-message" style="display:block;">Error: ${result.error}</p>`
                : `
                    <div class="image-panel"><img src="${result.image_url}" alt="Analyzed MRI Scan"></div>
                    <div class="results-panel">
                        <h3>${result.prediction.charAt(0).toUpperCase() + result.prediction.slice(1)}</h3>
                        <p class="confidence-score">File: ${result.filename} | Confidence: ${result.confidence}</p>
                        <p class="analysis-text">${result.analysis_text}</p>
                    </div>
                `;
            card.innerHTML = innerHTML;
            resultsContainer.appendChild(card);
        });
        showScreen('results');
    };

    showScreen('landing');
});
