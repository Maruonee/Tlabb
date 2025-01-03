    def save_to_csv(self, data_type, data):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{self.exp_date}_{self.exp_num}_accura"
        filename = os.path.join(self.folder_path, f"{folder_name}_{data_type}.csv")
        
        while True:
            try:
                file_exists = os.path.isfile(filename)
                with self.lock, open(filename, "a", newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        headers = ["Timestamp"] + list(data.keys())
                        writer.writerow(headers)
                    row = [timestamp] + list(data.values())
                    writer.writerow(row)
                break  # 저장 성공 시 루프 종료
            except PermissionError:
                print(f"Permission denied while accessing {filename}. Retrying...")
                time.sleep(0.1)  # 짧은 대기 후 재시도
            except Exception as e:
                print(f"Unexpected error while saving data: {e}")
                break  # 다른 예외 발생 시 루프 종료
