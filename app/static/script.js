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

    const resultImage = document.getElementById('result-image');
    const resultHeader = document.getElementById('result-header');
    const resultConfidence = document.getElementById('result-confidence');

    // --- State Management ---
    let currentScreen = 'landing';

    function showScreen(screenName) {
        for (const screen in screens) {
            screens[screen].classList.remove('active');
        }
        screens[screenName].classList.add('active');
        currentScreen = screenName;
    }

    // --- Event Listeners ---
    analyzeButton.addEventListener('click', () => showScreen('upload'));
    analyzeAnotherButton.addEventListener('click', () => {
        resetUploadUI();
        showScreen('upload');
    });

    dropZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        const files = e.target.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // Drag and Drop listeners
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('hover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('hover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('hover');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // --- File Handling and API Call ---
    function handleFile(file) {
        // Basic file type validation
        if (!file.type.startsWith('image/')) {
            showError('Please upload an image file (e.g., JPG, PNG).');
            return;
        }

        fileInfo.textContent = `Selected file: ${file.name}`;
        uploadError.style.display = 'none';

        // Show processing screen and start upload
        showScreen('processing');
        uploadFile(file);
    }

    async function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Server error');
            }

            const data = await response.json();
            displayResults(data, file);

        } catch (error) {
            showScreen('upload');
            showError(`Analysis failed: ${error.message}`);
        }
    }

    // --- UI Update Functions ---
    function displayResults(data, file) {
        // Set result text and color
        resultHeader.textContent = data.prediction === 'Pneumonia' ? 'Pneumonia Detected' : 'No Signs of Pneumonia';
        resultHeader.className = data.prediction === 'Pneumonia' ? 'positive' : 'negative';

        resultConfidence.textContent = `Confidence: ${data.confidence}`;

        // Display the uploaded image
        const reader = new FileReader();
        reader.onload = (e) => {
            resultImage.src = e.target.result;
        };
        reader.readAsDataURL(file);

        showScreen('results');
    }

    function showError(message) {
        uploadError.textContent = message;
        uploadError.style.display = 'block';
    }

    function resetUploadUI() {
        fileInfo.textContent = '';
        uploadError.style.display = 'none';
        fileInput.value = ''; // Reset file input
    }

    // --- Initial State ---
    showScreen('landing');
});
