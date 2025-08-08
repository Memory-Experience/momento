import { getHumeAccessToken } from "@/utils/getHumeAccessToken";
import Chat from "@/components/Chat";

export default async function Page() {
  const accessToken = await getHumeAccessToken();

  if (!accessToken) {
    throw new Error("Unable to get access token");
  }

  return (
    <div className={"grow flex flex-col"}>
      <Chat accessToken={accessToken} />
    </div>
  );
}
