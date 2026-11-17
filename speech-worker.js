const runtimeUrl = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1/dist/transformers.min.js';
const wasmUrl = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0-dev.20250409-89f8206ba4/dist/';
const model = 'Xenova/whisper-tiny';
const revision = '5332fcc35e32a33b86612b9a57a89be7906102b1';
const languages = { ru: 'russian', kk: 'kazakh', en: 'english' };
let recognizer;

self.onmessage = async ({ data: { audio, lang } }) => {
  try {
    if (!recognizer) {
      self.postMessage({ type: 'loading' });
      const { env, pipeline } = await import(runtimeUrl);
      env.allowLocalModels = false;
      env.backends.onnx.wasm.wasmPaths = wasmUrl;
      env.backends.onnx.wasm.numThreads = 1;
      env.backends.onnx.wasm.proxy = false;
      recognizer = await pipeline('automatic-speech-recognition', model, {
        revision,
        dtype: 'q8',
        device: 'wasm',
      });
    }
    self.postMessage({ type: 'recognizing' });
    const result = await recognizer(audio, {
      language: languages[lang] || languages.ru,
      task: 'transcribe',
      max_new_tokens: 160,
    });
    self.postMessage({ type: 'result', text: result.text.trim() });
  } catch (error) {
    console.warn('On-device voice recognition failed', error);
    self.postMessage({ type: 'error' });
  }
};
