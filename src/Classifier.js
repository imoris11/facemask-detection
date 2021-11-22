import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as knnClassifier from '@tensorflow-models/knn-classifier'
import './App.css';
import React, { useRef, useState } from 'react';
import FileReaderInput from 'react-file-reader-input';

function App() {
  const [maskedImages, setMaskedImages] = useState([]);
  const [noMaskedImages, setNoMaskedImages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [doneTraining, setDoneTraining] = useState(false);
  const [classifier, setClassifier] = useState(null);
  const [masked, setMasked] = useState(false);
  const camera = useRef();
  const figures = useRef();
  let net;
  /**
   * @type {knnClassifier}
   */

  const webcamElement = camera.current;
  console.log('Loading mobilenet..');

  const trainClassifier = async () => {
    // Train using mask images
    setLoading(true)
    let knn = knnClassifier.create();
    net = await mobilenet.load();
    const maskImages = document.querySelectorAll('.mask-img');
    maskImages.forEach(img => {
      const tfImg = tf.browser.fromPixels(img);
      const logits = net.infer(tfImg, 'conv_preds');
      knn.addExample(logits, 0); // has mask
    });
    // Train using no mask images
    const noMaskImages = document.querySelectorAll('.no-mask-img');
    noMaskImages.forEach(img => {
      const tfImg = tf.browser.fromPixels(img);
      const logits = net.infer(tfImg, 'conv_preds');
      knn.addExample(logits, 1); // no mask
    });
    setLoading(false)
    setDoneTraining(true);
    setClassifier(knn);
    console.log(knn.getNumClasses());
  }

  const run = async () => {
    net = await mobilenet.load();
    const webcam = await tf.data.webcam(webcamElement, {
      resizeHeight: 534,
      resizeWidth: 870
    });

    if (classifier) {
      console.log(classifier.getNumClasses())
    }

    while (true) {
      if (classifier && classifier.getNumClasses() > 0) {
        const img = await webcam.capture();
        // Get the activation from mobilenet from the webcam.
        const activation = net.infer(img, 'conv_preds');
        // Get the most likely class and confidence from the classifier module.
        const result = await classifier.predictClass(activation);
        if (figures.current) {
          //const classes = ['A', 'B', 'C'];
          const confidence = result.confidences[result.label]
          let text
          if (result.label === "0" && confidence >= 0.66) {
            text = "Mask Detected"
            setMasked(true)
          } else {
            text = "No Mask Detected"
            setMasked(false)
          }

          figures.current.innerText = `
        prediction: ${text} \n
        probability: ${result.confidences[result.label]}
        `;
          //figures.current.innerText = `Prediction: ${ result[0].className } \n probability: ${ result[0].probability } `;
        }
        img.dispose();
      }
      await tf.nextFrame();
    }
  };
  React.useEffect(() => {
    if (camera.current) {
      run();
    }

  }, [camera.current]);

  const handleChange = (e, results) => {
    const images = []
    results.forEach(result => {
      const [e, file] = result;
      images.push(e.target.result);
    });
    setMaskedImages(images)
  }

  const handleNoMask = (e, results) => {
    const images = []
    results.forEach(result => {
      const [e, file] = result;
      images.push(e.target.result);
    });
    setNoMaskedImages(images)
  }

  const handleToggle = () => {
    setMaskedImages([])
    setNoMaskedImages([])
  }

  return (
    <div className="App">
      <h3>Face Mask Detection</h3>
      <div ref={figures}></div>
      {doneTraining &&
        <video autoPlay playsInline muted={true}
          ref={camera}
          style={{ border: `10px solid ${masked ? 'green' : 'red'}` }}
          width="870" height="534"></video>}

        
      {maskedImages.length === 0 ?
      <FileReaderInput multiple id="my-file-input"
        onChange={handleChange}>
        <button className="upload-btn">Select Masked Photos!</button>
      </FileReaderInput> : null}
      
      {noMaskedImages.length === 0 ?
      <FileReaderInput multiple id="my-file-input"
        onChange={handleNoMask}>
        <button className="upload-btn">Select No Mask Photos!</button>
      </FileReaderInput> : null }

      {maskedImages.length > 0 ?
        <>
          <h3>Masked Images </h3>
          {maskedImages.map((img, idx) => <img key={`${idx}`} src={img} className="mask-img" />
          )}
        </> : null}
      {noMaskedImages.length > 0 ?
        <>
          <h3>No Masked Images </h3>
          {noMaskedImages.map((img, idx) =>
            <img src={img} key={`${idx}`} className="no-mask-img" />
          )}
        </> : null}

      {!doneTraining && maskedImages.length > 0 && noMaskedImages.length > 0 &&
        <div>
          {!loading ?
            <button className="upload-btn" onClick={() => trainClassifier()}>Train Model</button> : <p>Training model...</p>}
        </div>}
    </div>
  );
}

export default App;