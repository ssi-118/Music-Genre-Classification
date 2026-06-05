document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predictForm");
  const audioInput = document.getElementById("audioFile");
  const fileLabel = document.getElementById("fileLabel");
  const loadingMessage = document.getElementById("loadingMessage");
  const errorMessage = document.getElementById("errorMessage");
  const resultBox = document.getElementById("resultBox");
  const resultText = document.getElementById("resultText");
  const predictButton = form.querySelector("button[type='submit']");

  const allowedExtensions = ["au", "mp3", "wav"];

  function hideMessages() {
    loadingMessage.classList.add("hidden");
    errorMessage.classList.add("hidden");
    resultBox.classList.add("hidden");
    errorMessage.textContent = "";
  }

  function showError(message) {
    errorMessage.textContent = message;
    errorMessage.classList.remove("hidden");
  }

  audioInput.addEventListener("change", () => {
    const file = audioInput.files[0];

    if (file) {
      fileLabel.textContent = file.name;
      return;
    }

    fileLabel.textContent = "AU, MP3 or WAV, up to 25 MB";
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    hideMessages();

    const file = audioInput.files[0];
    if (!file) {
      showError("Please select an audio file before predicting.");
      return;
    }

    const extension = file.name.split(".").pop().toLowerCase();
    if (!allowedExtensions.includes(extension)) {
      showError("Unsupported file format. Please upload an MP3 or WAV file.");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    loadingMessage.classList.remove("hidden");
    predictButton.disabled = true;

    try {
      const response = await fetch("/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        showError(data.message || "Prediction failed. Please try again.");
        return;
      }

      resultText.textContent = data.genre;
      resultBox.classList.remove("hidden");
    } catch (error) {
      showError("Unable to reach the server. Please check that Flask is running.");
    } finally {
      loadingMessage.classList.add("hidden");
      predictButton.disabled = false;
    }
  });
});
