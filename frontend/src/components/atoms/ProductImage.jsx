import { useState } from "react";
import Avatar from "./Avatar";
import { C } from "../../styles/theme";

/**
 * Product photo with a graceful fallback.
 *
 * Images come from M2's image service, which does not hold a photo for every
 * article in the catalogue. A miss is expected rather than exceptional, so a
 * failed load swaps in the gradient placeholder instead of surfacing an error.
 */
export default function ProductImage({ src, alt, onClick }) {
  const [failed, setFailed] = useState(false);

  if (failed) {
    return (
      <Avatar size={80} radius={10} fontSize={26}>👗</Avatar>
    );
  }
  return (
    <img
      src={src}
      alt={alt}
      title="Click to enlarge"
      onError={() => setFailed(true)}
      onClick={onClick}
      style={{ width:80, height:80, borderRadius:10, flexShrink:0,
        objectFit:"cover", background:"#242424", cursor:"zoom-in",
        border:`1px solid ${C.border}` }}
    />
  );
}
