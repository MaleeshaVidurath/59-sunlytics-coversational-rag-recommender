import { useState } from "react";
import ProductImage from "../atoms/ProductImage";
import Avatar from "../atoms/Avatar";
import WhyList from "./WhyList";
import { M2_IMAGE_BASE } from "../../services/http";
import styles from "./ProductCard.module.css";

/** One recommended article: photo, name, colour/type/price, and its reasons. */
export default function ProductCard({ item, model }) {
  const [preview, setPreview] = useState(false);
  const imgSrc = `${M2_IMAGE_BASE}/api/images/${item.article_id}`;

  return (
    <div className={styles.card}>
      {item.article_id
        ? <ProductImage src={imgSrc} alt={item.name || "product"} onClick={() => setPreview(true)} />
        : <Avatar size={80} radius={10} fontSize={26}>👗</Avatar>}

      {preview && (
        <div className={styles.preview} onClick={() => setPreview(false)}>
          <img src={imgSrc} alt={item.name || "product"} className={styles.previewImage} />
          <div className={styles.previewName}>{item.name}</div>
          <div className={styles.previewAttributes}>
            {item.colour} · {item.type}{item.price ? ` · ${item.price}` : ""}
          </div>
          <div className={styles.previewClose}>Close</div>
        </div>
      )}

      <div className={styles.body}>
        <div className={styles.name}>{item.name}</div>
        <div className={styles.attributes}>
          {item.colour} · {item.type} · <span className={styles.price}>{item.price}</span>
        </div>
        {item.description && <div className={styles.description}>{item.description}</div>}
        <WhyList reasons={item.why || []} model={model} />
      </div>

      <div className={styles.meta}>
        <div className={styles.articleId}>#{item.article_id?.slice(-6)}</div>
        {/* Already a clamped 0-100 figure from the ranker; absent when nothing
            personalised matched, in which case no badge is shown at all. */}
        {typeof item.match_percent === "number" && (
          <div className={styles.matchPercent} title="Personalised match score">
            {item.match_percent}%
          </div>
        )}
      </div>
    </div>
  );
}
