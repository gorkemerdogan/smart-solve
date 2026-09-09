import { deployFullSmartSolve, writeDeploymentArtifact } from "./deploy";
import { verifyDeployment } from "./verify-deployment";

async function main() {
  const deployment = await deployFullSmartSolve();
  const { artifact, path } = await writeDeploymentArtifact(deployment);
  console.log("Deployment artifact:", path);
  await verifyDeployment(artifact);
}

if (require.main === module) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
