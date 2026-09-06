import { useDispatch } from "react-redux";
import CenteredTemplate from "../components/templates/CenteredTemplate";
import ModelGrid from "../components/organisms/ModelGrid";
import { modelSelected } from "../store/slices/modelSlice";

export default function ModelSelectPage() {
  const dispatch = useDispatch();
  return (
    <CenteredTemplate font="sans">
      <ModelGrid onSelect={model => dispatch(modelSelected(model))} />
    </CenteredTemplate>
  );
}
