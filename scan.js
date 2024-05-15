document.addEventListener("DOMContentLoaded", async function () {
  // Initialize TensorFlow
  await tf.setBackend("webgl");
  console.log("TensorFlow.js initialized");

  // Define variables
  let camView = "environment"; // Default camera view
  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  const fileInput = document.getElementById("fileInput");
  const toggleCamera = document.getElementById("toggleCamera");
  const captureButton = document.getElementById("captureButton");
  const galleryButton = document.getElementById("galleryButton");
  const overlay = document.getElementById("overlay");
  const closeButton = document.getElementById("closeButton");
  const bestMatchElement = document.getElementById("bestMatch");
  const confidenceElement = document.getElementById("confidence");
  const scannedImageElement = document.getElementById("scannedImage");
  const imageNames = [
    "almond Plant",
    "almonddry Plant",
    "almondleaf Plant",
    "avocado Plant",
    "avocadodry Plant",
    "avocadoleaf Plant",
    "cashew Plant",
    "cashewdry Plant",
    "cashewleaf Plant",
    "guava Plant",
    "guavadry Plant",
    "guavaleaf Plant",
    "mango Plant",
    "mangodry Plant",
    "mangoleaf Plant",
  ];

  // Load MobileNet model
  const net = await mobilenet.load();
  console.log("MobileNet model loaded");

  // Create KNN classifier
  const classifier = knnClassifier.create();

  // Function to load image data to ImageData object
  const loadImageToImageData = async (imageUrl) => {
    // Fetch the image file as binary data
    const response = await fetch(imageUrl);
    const blob = await response.blob();
    const arrayBuffer = await blob.arrayBuffer();

    // Create an image element
    const img = document.createElement("img");

    // Set up a promise to resolve when the image is loaded
    const imageLoaded = new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
    });

    // Set the src attribute to the image data
    img.src = URL.createObjectURL(new Blob([arrayBuffer]));

    // Wait for the image to be loaded
    await imageLoaded;

    // Create a canvas element and draw the image onto it
    const ctx = canvas.getContext("2d");
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);

    // Get the ImageData object
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    // Return the ImageData object
    return imageData;
  };

  // Function to compare image with dataset
  const compareWithDataset = async (uploadedImage, imageUrl) => {
    // Define dataset containing paths to images
    const dataset = [
      "./labels/almond.png",
      "./labels/almonddry.png",
      "./labels/almondleaf.png",
      "./labels/avocado.png",
      "./labels/avocadodry.png",
      "./labels/avocadoleaf.png",
      "./labels/cashew.png",
      "./labels/cashewdry.png",
      "./labels/cashewleaf.png",
      "./labels/guava.png",
      "./labels/guavadry.png",
      "./labels/guavaleaf.png",
      "./labels/mango.png",
      "./labels/mangodry.png",
      "./labels/mangoleaf.png",
    ];

    for (let i = 0; i < dataset.length; i++) {
      const imageData = await loadImageToImageData(dataset[i]);
      const activation = net.infer(imageData, true);
      console.log(activation);
      classifier.addExample(activation, i);
    }

    if (classifier.getNumClasses() > 0) {
      const activation = net.infer(uploadedImage, true);
      const result = await classifier.predictClass(activation);
      console.log(result);

      bestMatchElement.textContent = imageNames[result.label];
      confidenceElement.textContent = result.confidences[result.label];

      // Display scanned image
      scannedImageElement.innerHTML = `<img src="${imageUrl}" width="150" height="150" alt="">`;
    }

    overlay.style.display = "flex";
  };

  //Click on gallery button to select image
  galleryButton.addEventListener("click", function () {
    fileInput.click(); // Trigger click on file input
  });

  // Function to handle file input change
  fileInput.addEventListener("change", async (event) => {
    const file = event.target.files[0];
    const imageUrl = await readFileAsDataURL(file);
    const img = await createImageBitmap(file);
    const ctx = canvas.getContext("2d");
    canvas.width = img.width;
    canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    compareWithDataset(imageData, imageUrl);
  });

  // Function to handle camera capture
  captureButton.addEventListener("click", async () => {
    navigator.mediaDevices
      .getUserMedia({
        video: { facingMode: "environment" }, // Use the rear camera
        audio: false,
      })
      .then(function (stream) {
        video.srcObject = stream;
        video.addEventListener("loadedmetadata", async function () {
          const canvas = document.createElement("canvas");
          const ctx = canvas.getContext("2d");
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

          // Convert canvas to TensorFlow.js tensor or other compatible format
          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          // Process imageData with TensorFlow.js
          console.log(imageData);

          canvas.toBlob(async (blob) => {
            // Use the blob as needed
            console.log(blob);
            const imageUrl = await readFileAsDataURL(blob);
            compareWithDataset(imageData, imageUrl);
          }, "image/jpeg");
        });
        video.play();
      })
      .catch(function (error) {
        console.error("Error accessing camera:", error);
      });

    // const webcam = await tf.data.webcam(video);
    // const img = await webcam.capture();
    // // Convert the captured image to a canvas
    // const canvas = document.createElement("canvas");
    // canvas.width = img.shape[1];
    // canvas.height = img.shape[0];
    // const ctx = canvas.getContext("2d");
    // await tf.browser.toPixels(img, canvas);

    // // Convert the canvas to a Blob
    // canvas.toBlob(async (blob) => {
    //   // Use the blob as needed
    //   console.log(blob);
    //   const imageUrl = await readFileAsDataURL(blob);
    //   const converted = tensorToImageData(img);
    //   compareWithDataset(converted, imageUrl);
    // }, "image/jpeg");
  });

  function readFileAsDataURL(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        resolve(reader.result);
      };
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  const tensorToImageData = (tensor) => {
    // Get the data from the TensorFlow tensor
    const data = tensor.dataSync();
    // Get the dimensions of the image
    const [height, width, channels] = tensor.shape;
    // Create a canvas element
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    // Create an empty ImageData object
    const upImg = ctx.createImageData(width, height);
    // Fill the ImageData with the pixel data from the TensorFlow tensor
    for (let i = 0; i < data.length; i++) {
      upImg.data[i] = data[i];
    }
    // Draw the ImageData onto the canvas
    ctx.putImageData(upImg, 0, 0);
    // Return the ImageData object
    return upImg;
  };

  // Function to handle camera toggle
  toggleCamera.addEventListener("click", () => {
    camView = camView === "environment" ? "user" : "environment";
    startWebcam();
  });

  // Function to start webcam
  async function startWebcam() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false,
      });
      video.srcObject = stream;
    } catch (error) {
      console.error(error);
      console.log("Attempting to use front camera...");
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "environment" },
          audio: false,
        });
        video.srcObject = stream;
      } catch (error) {
        console.error("Error accessing camera:", error);
      }
    }
  }

  // Function to close overlay
  closeButton.addEventListener("click", () => {
    overlay.style.display = "none";
  });

  // Start webcam initially
  await startWebcam();
});
