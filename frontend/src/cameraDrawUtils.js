/**
 * Source rectangle for CSS object-fit: cover — crop `video` to fill a
 * destination with aspect ratio destW / destH.
 */
export function getCoverSourceRect(videoWidth, videoHeight, destW, destH) {
  const vw = videoWidth || 1;
  const vh = videoHeight || 1;
  const destAspect = destW / destH;
  const srcAspect = vw / vh;
  let sx;
  let sy;
  let sw;
  let sh;
  if (srcAspect > destAspect) {
    sh = vh;
    sw = destAspect * vh;
    sx = (vw - sw) / 2;
    sy = 0;
  } else {
    sw = vw;
    sh = vw / destAspect;
    sx = 0;
    sy = (vh - sh) / 2;
  }
  return { sx, sy, sw, sh };
}
