import subprocess

class VMwareBlacklist:
    def __init__(self):
        self.blacklist = []

    def add_to_blacklist(self, ip_address):
        self.blacklist.append(ip_address)

    def remove_from_blacklist(self, ip_address):
        if ip_address in self.blacklist:
            self.blacklist.remove(ip_address)

    def update_vmware_blacklist(self):
        # Run the command to update the VMware blacklist
        command = ["vmware-cli", "update-blacklist"]
        for ip_address in self.blacklist:
            command.extend(["--add", ip_address])

        subprocess.run(command)
        print("VMware blacklist updated successfully.")

if __name__ == '__main__':
    vmware_blacklist = VMwareBlacklist()

    # Add IP addresses to the blacklist
    vmware_blacklist.add_to_blacklist("192.168.1.100")
    vmware_blacklist.add_to_blacklist("192.168.1.101")
    vmware_blacklist.add_to_blacklist("192.168.1.102")

    # Remove an IP address from the blacklist
    vmware_blacklist.remove_from_blacklist("192.168.1.101")

    # Update the VMware blacklist
    vmware_blacklist.update_vmware_blacklist()
