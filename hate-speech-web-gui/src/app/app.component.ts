import { HttpClient } from '@angular/common/http';
import { Component } from '@angular/core';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = 'hate-speech-web-gui';

  textToCheck: string = '';
  result: string | null = null;

  constructor(private http: HttpClient) {}

  onSubmit() {
    this.http.post<any>('/predict', { text: this.textToCheck }).subscribe({
      next: (response) => {
        this.result = response.result;
      },
      error: (error) => {
        console.error('Error:', error);
      },
      complete: () => console.log('Request completed') // Optional
    });
  }
}
