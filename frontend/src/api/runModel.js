import { runTrainRequest } from './train';
import { runCrossvalRequest } from './cv';

// Simple router: use holdout -> /train, kfold -> /cv
export async function runModelRequest(payload) {
  const mode = payload?.split?.mode;
  if (mode === 'holdout') {
    return runTrainRequest(payload);
  }
  if (mode === 'kfold') {
    return runCrossvalRequest(payload);
  }
  throw new Error(`Unsupported split mode: ${mode}`);
}
