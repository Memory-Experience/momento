import Image from "next/image";
import Link from "next/link";
import { FC } from "react";

const Header: FC = () => {
  return (
    <header className="p-4 px-12 shadow-md">
      <Link href="/" className="items-center">
        <Image src="/logo.png" alt="Momento Logo" width={122} height={30} />
      </Link>
    </header>
  );
};

export default Header;
